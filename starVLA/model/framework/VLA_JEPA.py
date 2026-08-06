# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer, VJEPA2VideoProcessor

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import (
    VisionTransformerPredictorAC,
    VisionTransformerPredictorAC_New,
)
from starVLA.model.modules.world_model.causal_adaln_predictor import (
    balanced_diagonal_gaussian_kl,
    CausalAdaLNWorldModel,
    GaussianCodeProjector,
    split_video_into_tubelets,
)
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        qwenvl_config = self.config.framework.get("qwenvl", {})
        self.qwen_input_mode = str(qwenvl_config.get("input_mode", "image")).lower()
        if self.qwen_input_mode not in {"image", "video", "paired_image_video"}:
            raise ValueError(
                f"Unsupported Qwen input_mode={self.qwen_input_mode!r}; "
                "expected 'image', 'video', or 'paired_image_video'."
            )
        self.qwen_visual_resized_height = int(
            qwenvl_config.get("visual_resized_height", 256)
        )
        self.qwen_visual_resized_width = int(
            qwenvl_config.get("visual_resized_width", 256)
        )
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )

        # TODO speical tokens

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        # The official VLA-JEPA YAML stores this under ``framework.action_model``.
        # Upstream accidentally reads ``trainer.repeated_diffusion_steps`` in
        # forward(), which silently falls back to 4 even though the released
        # config requests 8. Prefer the documented action-model field while
        # retaining the trainer key only as a compatibility fallback.
        self.repeated_diffusion_steps = int(
            config.framework.action_model.get(
                "repeated_diffusion_steps",
                config.trainer.get("repeated_diffusion_steps", 4),
            )
        )
        if self.repeated_diffusion_steps < 1:
            raise ValueError("repeated_diffusion_steps must be at least 1.")
        
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        tubelet_size = self.vj_encoder.config.tubelet_size
        self.vj_predictor_variant = str(
            self.config.framework.vj2_model.get("predictor_variant", "original")
        )
        predictor_classes = {
            "original": VisionTransformerPredictorAC,
            "f0_repeat_noncausal": VisionTransformerPredictorAC_New,
        }
        self.is_causal_kl_variant = (
            self.vj_predictor_variant == "causal_adaln_tubelet"
        )
        supported_predictors = tuple(predictor_classes) + (
            "causal_adaln_tubelet",
        )
        if self.vj_predictor_variant not in supported_predictors:
            raise ValueError(
                f"Unsupported V-JEPA predictor_variant={self.vj_predictor_variant!r}; "
                f"expected one of {supported_predictors}."
            )

        vj2_config = self.config.framework.vj2_model
        num_vj_views = int(vj2_config.get("num_views", 2))
        vj_embed_dim = self.vj_encoder.config.hidden_size * num_vj_views
        if self.is_causal_kl_variant:
            if self.qwen_input_mode != "paired_image_video":
                raise ValueError(
                    "causal_adaln_tubelet requires "
                    "qwenvl.input_mode=paired_image_video."
                )
            encoder_tubelet_size = int(self.vj_encoder.config.tubelet_size)
            num_frames = int(vj2_config.num_frames)
            if num_frames % encoder_tubelet_size:
                raise ValueError(
                    f"num_frames={num_frames} must be divisible by the frozen "
                    f"encoder tubelet_size={encoder_tubelet_size}."
                )
            self.num_target_tubelets = num_frames // encoder_tubelet_size
            self.rollout_target_tubelet_indices = tuple(
                int(index)
                for index in vj2_config.get(
                    "rollout_target_tubelet_indices",
                    list(range(1, self.num_target_tubelets)),
                )
            )
            if not self.rollout_target_tubelet_indices:
                raise ValueError("rollout_target_tubelet_indices cannot be empty.")
            if tuple(sorted(set(self.rollout_target_tubelet_indices))) != (
                self.rollout_target_tubelet_indices
            ):
                raise ValueError(
                    "rollout_target_tubelet_indices must be unique and increasing."
                )
            if self.rollout_target_tubelet_indices[0] <= 0:
                raise ValueError(
                    "Rollout targets must follow the initial 0th tubelet."
                )
            if (
                self.rollout_target_tubelet_indices[-1]
                >= self.num_target_tubelets
            ):
                raise ValueError(
                    "rollout_target_tubelet_indices exceed the configured "
                    "video horizon."
                )
            expected_tubelet_indices = tuple(
                range(1, self.num_target_tubelets)
            )
            if (
                self.rollout_target_tubelet_indices
                != expected_tubelet_indices
            ):
                raise ValueError(
                    "Teacher-forced tubelet training requires every shifted "
                    "target in order; expected "
                    f"{expected_tubelet_indices}, got "
                    f"{self.rollout_target_tubelet_indices}."
                )

            latent_config = self.config.framework.get("latent_alignment", {})
            self.latent_dim = int(latent_config.get("latent_dim", 256))
            self.kl_weight = float(latent_config.get("kl_weight", 0.01))
            self.kl_free_bits = float(latent_config.get("free_bits", 0.0))
            self.kl_dynamics_scale = float(
                latent_config.get("dynamics_scale", 1.0)
            )
            self.kl_representation_scale = float(
                latent_config.get("representation_scale", 0.1)
            )
            self.posterior_prompt = str(
                latent_config.get(
                    "posterior_prompt",
                    "Infer the temporal dynamics from this video {actions}.",
                )
            )
            self.posterior_use_instruction = bool(
                latent_config.get("posterior_use_instruction", False)
            )
            self.deterministic_latent_eval = bool(
                latent_config.get("deterministic_eval", True)
            )
            self.vla_wm_loss_weight = float(
                vj2_config.get("vla_wm_loss_weight", 0.1)
            )
            self.vlm_wm_loss_weight = float(
                vj2_config.get("vlm_wm_loss_weight", 1.0)
            )
            self.tubelet_encode_batch_size = int(
                vj2_config.get("tubelet_encode_batch_size", 64)
            )
            if (
                self.kl_weight < 0
                or self.kl_free_bits < 0
                or self.kl_dynamics_scale < 0
                or self.kl_representation_scale < 0
                or self.vla_wm_loss_weight < 0
                or self.vlm_wm_loss_weight < 0
            ):
                raise ValueError(
                    "KL/free-bits/balance and WM loss weights must be "
                    "non-negative."
                )
            if self.tubelet_encode_batch_size <= 0:
                raise ValueError("tubelet_encode_batch_size must be positive.")

            self.code_projector = GaussianCodeProjector(
                input_dim=self.qwen_vl_interface.model.config.hidden_size,
                latent_dim=self.latent_dim,
                min_logvar=float(latent_config.get("min_logvar", -6.0)),
                max_logvar=float(latent_config.get("max_logvar", 2.0)),
            )
            encoder_patch_size = int(self.vj_encoder.config.patch_size)
            configured_patch_size = int(
                vj2_config.get("patch_size", encoder_patch_size)
            )
            if configured_patch_size != encoder_patch_size:
                raise ValueError(
                    "The causal WM patch grid must match the frozen V-JEPA 2 "
                    f"encoder: configured {configured_patch_size}, encoder "
                    f"{encoder_patch_size}."
                )
            self.vj_predictor = CausalAdaLNWorldModel(
                img_size=(
                    self.vj_encoder.config.image_size,
                    self.vj_encoder.config.image_size,
                ),
                patch_size=encoder_patch_size,
                embed_dim=vj_embed_dim,
                predictor_embed_dim=int(
                    vj2_config.get("predictor_embed_dim", 1024)
                ),
                latent_dim=self.latent_dim,
                code_tokens_per_step=int(
                    vj2_config.num_action_tokens_per_timestep
                ),
                depth=int(vj2_config.depth),
                num_heads=int(vj2_config.num_heads),
                mlp_ratio=float(vj2_config.get("mlp_ratio", 4.0)),
                drop_rate=float(vj2_config.get("drop_rate", 0.0)),
                attn_drop_rate=float(vj2_config.get("attn_drop_rate", 0.0)),
                drop_path_rate=float(vj2_config.get("drop_path_rate", 0.0)),
                use_silu=bool(vj2_config.get("use_silu", False)),
                wide_silu=bool(vj2_config.get("wide_silu", True)),
                use_activation_checkpointing=bool(
                    vj2_config.get("use_activation_checkpointing", True)
                ),
            )
            # This encoder is a fixed target network, independent of trainer
            # freeze strings. Keeping it in eval mode also disables stochastic
            # depth when the parent model enters train mode.
            self.vj_encoder.requires_grad_(False)
            self.vj_encoder.eval()
            num_code_groups = len(self.rollout_target_tubelet_indices)
        else:
            predictor_cls = predictor_classes[self.vj_predictor_variant]
            self.vj_predictor = predictor_cls(
                num_frames=self.config.framework.vj2_model.num_frames//tubelet_size,
                img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
                tubelet_size=1,
                depth=self.config.framework.vj2_model.depth,
                num_heads=self.config.framework.vj2_model.num_heads,
                embed_dim=vj_embed_dim,
                action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
                num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
            )
            num_vj_states = self.config.framework.vj2_model.num_frames // tubelet_size
            num_code_groups = num_vj_states if self.vj_predictor_variant == "f0_repeat_noncausal" else num_vj_states - 1
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:num_code_groups]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])

    def expand_tokenizer(self, 
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            # 2) resize embeddings of vla
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def train(self, mode: bool = True):
        """Keep the causal variant's target encoder frozen and deterministic."""

        super().train(mode)
        if getattr(self, "is_causal_kl_variant", False):
            self.vj_encoder.eval()
        return self

    def _extract_qwen_hidden_tokens(
        self,
        qwen_inputs,
        *,
        expected_action_tokens: int,
        extract_embodied_tokens: bool,
    ):
        """Run Qwen once and gather the special-token hidden states."""

        action_ids = torch.tensor(
            self.action_token_ids,
            device=qwen_inputs["input_ids"].device,
        )
        action_mask = torch.isin(qwen_inputs["input_ids"], action_ids)
        action_counts = action_mask.sum(dim=1)
        if not torch.all(action_counts == expected_action_tokens):
            raise ValueError(
                "Every Qwen route must contain exactly "
                f"{expected_action_tokens} latent action tokens; got "
                f"{action_counts.detach().cpu().tolist()}."
            )

        embodied_mask = None
        if extract_embodied_tokens:
            embodied_mask = qwen_inputs["input_ids"].eq(self.embodied_action_token_id)
            expected_embodied_tokens = int(
                self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction
            )
            embodied_counts = embodied_mask.sum(dim=1)
            if not torch.all(embodied_counts == expected_embodied_tokens):
                raise ValueError(
                    "Every robot prior must contain exactly "
                    f"{expected_embodied_tokens} embodied action tokens; got "
                    f"{embodied_counts.detach().cpu().tolist()}."
                )

        qwen_outputs = self.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = qwen_outputs.hidden_states[-1]
        batch_size, _, hidden_dim = last_hidden.shape
        action_tokens = last_hidden[action_mask].view(
            batch_size,
            expected_action_tokens,
            hidden_dim,
        )
        embodied_tokens = None
        if embodied_mask is not None:
            embodied_tokens = last_hidden[embodied_mask].view(
                batch_size,
                -1,
                hidden_dim,
            )
        return action_tokens, embodied_tokens

    def _prepare_vj_videos(self, batch_videos: np.ndarray) -> torch.Tensor:
        """Apply the official Hugging Face V-JEPA 2 preprocessing per view."""

        if batch_videos.ndim != 6:
            raise ValueError(
                "Expected videos [B,V,T,H,W,C], got "
                f"{tuple(batch_videos.shape)}."
            )
        batch_size, num_views, _, _, _, channels = batch_videos.shape
        if channels != 3:
            raise ValueError("V-JEPA videos must be RGB.")

        videos_chw = batch_videos.transpose(0, 1, 2, 5, 3, 4)
        videos_chw = videos_chw.reshape(
            batch_size * num_views,
            *videos_chw.shape[2:],
        )
        encoder_device = next(self.vj_encoder.parameters()).device
        processed = [
            self.vj_processor(videos=video, return_tensors="pt")[
                "pixel_values_videos"
            ]
            for video in videos_chw
        ]
        return torch.cat(processed, dim=0).to(encoder_device)

    def _encode_vj_tubelets_independently(
        self,
        input_videos: torch.Tensor,
        *,
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor:
        """Encode each non-overlapping tubelet with the frozen V-JEPA 2 encoder.

        For the released encoder, ``tubelet_size=2``. An eight-frame input is
        therefore split into four isolated clips: ``01``, ``23``, ``45``, and
        ``67``. Encoding those clips separately preserves the official temporal
        patch embedding while preventing target leakage across rollout states.
        Output is restored to ``[B,S,P,V*D]``, where ``S=T/tubelet_size``.
        """

        if input_videos.ndim != 5:
            raise ValueError(
                "Expected preprocessed videos [B*V,T,C,H,W], got "
                f"{tuple(input_videos.shape)}."
            )
        if input_videos.shape[0] != batch_size * num_views:
            raise ValueError("Preprocessed V-JEPA batch/view dimensions do not match.")
        num_frames = input_videos.shape[1]
        expected_frames = int(self.config.framework.vj2_model.num_frames)
        if num_frames != expected_frames:
            raise ValueError(f"Expected {expected_frames} frames, got {num_frames}.")

        tubelet_size = int(self.vj_encoder.config.tubelet_size)
        if num_frames % tubelet_size:
            raise ValueError(
                f"{num_frames} frames cannot be split into tubelets of "
                f"size {tubelet_size}."
            )
        num_tubelets = num_frames // tubelet_size
        tubelet_clips = split_video_into_tubelets(
            input_videos,
            batch_size=batch_size,
            num_views=num_views,
            tubelet_size=tubelet_size,
        )

        encoded_chunks = []
        self.vj_encoder.eval()
        with torch.no_grad():
            for start in range(
                0,
                tubelet_clips.shape[0],
                self.tubelet_encode_batch_size,
            ):
                encoded = self.vj_encoder.get_vision_features(
                    pixel_values_videos=tubelet_clips[
                        start : start + self.tubelet_encode_batch_size
                    ]
                )
                encoded_chunks.append(encoded.detach())
        tubelet_features = torch.cat(encoded_chunks, dim=0)
        if tubelet_features.shape[1] != self.vj_predictor.tokens_per_frame:
            raise ValueError(
                "Frozen encoder/predictor patch-grid mismatch: encoder returned "
                f"{tubelet_features.shape[1]} tokens, predictor expects "
                f"{self.vj_predictor.tokens_per_frame}."
            )

        feature_dim = tubelet_features.shape[-1]
        tubelet_features = tubelet_features.view(
            batch_size,
            num_views,
            num_tubelets,
            self.vj_predictor.tokens_per_frame,
            feature_dim,
        )
        # Preserve batch/view ordering before concatenating the view channels.
        tubelet_features = tubelet_features.permute(0, 2, 3, 1, 4).reshape(
            batch_size,
            num_tubelets,
            self.vj_predictor.tokens_per_frame,
            num_views * feature_dim,
        )
        return tubelet_features.detach()

    def _compute_action_loss(
        self,
        *,
        actions,
        embodied_action_tokens: torch.Tensor,
        state,
    ) -> torch.Tensor:
        actions_tensor = torch.tensor(
            np.array(actions),
            device=embodied_action_tokens.device,
            dtype=embodied_action_tokens.dtype,
        )
        actions_target = actions_tensor[
            :,
            -(self.future_action_window_size + 1) :,
            :,
        ]
        repeated_diffusion_steps = self.repeated_diffusion_steps
        actions_target = actions_target.repeat(repeated_diffusion_steps, 1, 1)
        embodied_action_tokens = embodied_action_tokens.repeat(
            repeated_diffusion_steps,
            1,
            1,
        )

        state_repeated = None
        if state is not None:
            state_repeated = torch.tensor(
                np.array(state),
                device=embodied_action_tokens.device,
                dtype=embodied_action_tokens.dtype,
            ).repeat(repeated_diffusion_steps, 1, 1)
        return self.action_model(
            embodied_action_tokens,
            actions_target,
            state_repeated,
        )

    def _forward_causal_kl(
        self,
        *,
        examples,
        batch_images,
        batch_videos,
        instructions,
        actions,
        state,
        compute_zero_code_metric: bool,
    ):
        """Paired posterior/prior path with a frozen target and causal rollout."""

        batch_size, num_views, _ = batch_videos.shape[:3]
        expected_views = int(self.config.framework.vj2_model.get("num_views", 2))
        if num_views != expected_views:
            raise ValueError(
                f"Expected {expected_views} V-JEPA views, got {num_views}."
            )
        rollout_steps = len(self.rollout_target_tubelet_indices)
        code_tokens_per_step = int(
            self.config.framework.vj2_model.num_action_tokens_per_timestep
        )
        expected_action_tokens = rollout_steps * code_tokens_per_step

        # Compute the frozen target first. This releases V-JEPA 2's temporary
        # activations before either trainable Qwen graph is constructed.
        input_videos = self._prepare_vj_videos(batch_videos)
        target_features = self._encode_vj_tubelets_independently(
            input_videos,
            batch_size=batch_size,
            num_views=num_views,
        )
        del input_videos

        # Image + language is the inference-time prior.
        dataset_config = (
            self.config.datasets.vla_data
            if actions is not None
            else self.config.datasets.video_data
        )
        prior_replacements = {"{actions}": self.replace_prompt}
        if actions is not None:
            prior_replacements["{e_actions}"] = self.embodied_replace_prompt
        prior_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict=prior_replacements,
            prompt_template=dataset_config.get("CoT_prompt", ""),
            visual_resized_height=self.qwen_visual_resized_height,
            visual_resized_width=self.qwen_visual_resized_width,
        )
        prior_tokens, embodied_action_tokens = self._extract_qwen_hidden_tokens(
            prior_inputs,
            expected_action_tokens=expected_action_tokens,
            extract_embodied_tokens=actions is not None,
        )

        # Future video is the training-only posterior. Qwen consumes one video
        # stream while the frozen target encoder retains both configured views.
        missing_fps = [
            index for index, example in enumerate(examples) if "video_fps" not in example
        ]
        if missing_fps:
            raise ValueError(
                "Paired video posterior requires dataset-provided video_fps; "
                f"missing for samples {missing_fps}."
            )
        latent_config = self.config.framework.latent_alignment
        posterior_view_index = int(
            latent_config.get("posterior_video_view_index", 0)
        )
        if not 0 <= posterior_view_index < num_views:
            raise ValueError(
                f"posterior_video_view_index={posterior_view_index} is out of range "
                f"for {num_views} views."
            )
        posterior_videos = [
            [
                frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
                for frame in example["video"][posterior_view_index]
            ]
            for example in examples
        ]
        posterior_instructions = (
            instructions if self.posterior_use_instruction else [""] * batch_size
        )
        posterior_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            videos=posterior_videos,
            video_fps=[float(example["video_fps"]) for example in examples],
            instructions=posterior_instructions,
            prompt_replace_dict={"{actions}": self.replace_prompt},
            prompt_template=self.posterior_prompt,
            visual_resized_height=self.qwen_visual_resized_height,
            visual_resized_width=self.qwen_visual_resized_width,
        )
        posterior_tokens, _ = self._extract_qwen_hidden_tokens(
            posterior_inputs,
            expected_action_tokens=expected_action_tokens,
            extract_embodied_tokens=False,
        )

        prior_mean, prior_logvar = self.code_projector(prior_tokens)
        posterior_mean, posterior_logvar = self.code_projector(posterior_tokens)
        (
            kl_objective,
            dynamics_kl,
            representation_kl,
            raw_kl,
        ) = balanced_diagonal_gaussian_kl(
            posterior_mean,
            posterior_logvar,
            prior_mean,
            prior_logvar,
            free_bits=self.kl_free_bits,
            dynamics_scale=self.kl_dynamics_scale,
            representation_scale=self.kl_representation_scale,
        )
        posterior_sample = self.code_projector.sample(
            posterior_mean,
            posterior_logvar,
            deterministic=(not self.training and self.deterministic_latent_eval),
        )
        posterior_codes = posterior_sample.view(
            batch_size,
            rollout_steps,
            code_tokens_per_step,
            self.latent_dim,
        )

        rollout_targets = target_features[
            :,
            self.rollout_target_tubelet_indices,
        ]
        teacher_forcing_contexts = target_features[:, :-1]
        if self.training:
            predicted_features = self.vj_predictor.teacher_forced(
                teacher_forcing_contexts,
                posterior_codes,
            )
        else:
            predicted_features = self.vj_predictor.rollout(
                target_features[:, 0],
                posterior_codes,
            )
        per_step_wm_l1 = (
            predicted_features.float() - rollout_targets.float()
        ).abs().mean(dim=(0, 2, 3))
        wm_l1 = per_step_wm_l1.mean()

        diagnostic_metrics = {
            "kl_raw_metric": raw_kl.detach(),
            "kl_dynamics_metric": dynamics_kl.detach(),
            "kl_representation_metric": representation_kl.detach(),
            "posterior_std_metric": torch.exp(
                0.5 * posterior_logvar.float()
            ).mean().detach(),
            "prior_std_metric": torch.exp(
                0.5 * prior_logvar.float()
            ).mean().detach(),
            "posterior_prior_mean_l1_metric": (
                posterior_mean.float() - prior_mean.float()
            ).abs().mean().detach(),
        }
        for step_index, step_l1 in enumerate(per_step_wm_l1, start=1):
            diagnostic_metrics[f"wm_rollout_step_{step_index}_l1_metric"] = (
                step_l1.detach()
            )

        if compute_zero_code_metric:
            with torch.no_grad():
                if self.training:
                    zero_predictions = self.vj_predictor.teacher_forced(
                        teacher_forcing_contexts,
                        torch.zeros_like(posterior_codes),
                    )
                else:
                    zero_predictions = self.vj_predictor.rollout(
                        target_features[:, 0],
                        torch.zeros_like(posterior_codes),
                    )
                zero_code_wm_l1 = (
                    zero_predictions.float() - rollout_targets.float()
                ).abs().mean()
            diagnostic_metrics.update(
                {
                    "normal_code_wm_l1_metric": wm_l1.detach(),
                    "zero_code_wm_l1_metric": zero_code_wm_l1.detach(),
                }
            )

        output = {
            "wm_loss": wm_l1
            * (
                self.vla_wm_loss_weight
                if actions is not None
                else self.vlm_wm_loss_weight
            ),
            "kl_loss": kl_objective * self.kl_weight,
            **diagnostic_metrics,
        }
        if actions is not None:
            if embodied_action_tokens is None:
                raise RuntimeError("Robot prior did not return embodied action tokens.")
            with torch.autocast("cuda", dtype=torch.float32):
                output["action_loss"] = self._compute_action_loss(
                    actions=actions,
                    embodied_action_tokens=embodied_action_tokens,
                    state=state,
                )
        return output

    def forward(
        self,
        examples: List[dict] = None,
        compute_zero_code_metric: bool = False,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
        batch_videos = [example["video"] for example in examples]  #  [B, V, T, H, W, 3]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"]for example in examples] if "action" in examples[0] else None # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        if self.is_causal_kl_variant:
            return self._forward_causal_kl(
                examples=examples,
                batch_images=batch_images,
                batch_videos=np.stack(batch_videos),
                instructions=instructions,
                actions=actions,
                state=state,
                compute_zero_code_metric=compute_zero_code_metric,
            )

        qwen_video_kwargs = {}
        if self.qwen_input_mode == "video":
            missing_fps = [
                sample_index
                for sample_index, example in enumerate(examples)
                if "video_fps" not in example
            ]
            if missing_fps:
                raise ValueError(
                    "Qwen video input requires dataset-provided video_fps; "
                    f"missing for samples {missing_fps}."
                )

            # SSV2 currently creates two identical V-JEPA views. Qwen needs one
            # temporal stream, so pass the first view's full eight-frame clip
            # instead of duplicating the same video in its context.
            qwen_videos = [
                [
                    frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
                    for frame in example["video"][0]
                ]
                for example in examples
            ]
            qwen_video_kwargs = {
                "videos": qwen_videos,
                "video_fps": [float(example["video_fps"]) for example in examples],
                "visual_resized_height": self.qwen_visual_resized_height,
                "visual_resized_width": self.qwen_visual_resized_width,
            }

        """
        if self.action_model.device == torch.device("cuda:0") and "action" in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(actions[0].shape) # [T-1, action_dim]
            print(state[0].shape) if state is not None else print("No state") #[state_dim]
            print(len(batch_videos), len(instructions), len(actions), len(state) if state is not None else "No state")
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "data_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "data_view_1.mp4")
            batch_images[0][0].save("data_image_view_0.png")
            batch_images[0][1].save("data_image_view_1.png")
            #print(self.action_tokens)
            print(self.replace_prompt)
            print(self.action_token_ids)
        elif self.action_model.device == torch.device("cuda:0") and "action" not in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(len(batch_videos), len(instructions))
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "video_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "video_view_1.mp4")
            batch_images[0][0].save("video_image_view_0.png")
        exit()
        """
        
        

        #[print(each.shape, end=";") for each in batch_videos]
        batch_videos = np.stack(batch_videos)  #  [B, V, T, H, W, 3]
        batch_videos = batch_videos.transpose(0,1,2,5,3,4)  # [B, V, T, 3, H, W]

        # Step 1: QWenVL input format
        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
                **qwen_video_kwargs,
            )
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""),
                **qwen_video_kwargs,
            )
        
        action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        action_indices = action_indices.nonzero(as_tuple=True)

        # TODO action condition tokens
        #embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            #print(action_tokens.shape, last_hidden.shape, embodied_action_tokens.shape)
            #exit()
        
            # Step 2: JEPA Encoder
            B, V, T, C, H, W = batch_videos.shape
            batch_videos = batch_videos.reshape(B*V, T, C, H, W)  # [B*V, T, C, H, W]
            input_videos = []
            for i in range(B*V):
                input_videos.append(self.vj_processor(
                    videos=batch_videos[i], return_tensors="pt"
                )["pixel_values_videos"].to(self.vj_encoder.device))
            input_videos = torch.cat(input_videos, dim=0)  # [B*V, T, C, H, W]
            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)
                if self.vj_predictor_variant == "f0_repeat_noncausal":
                    current_videos = input_videos[:, :1].repeat(
                        1,
                        self.vj_encoder.config.tubelet_size,
                        1,
                        1,
                        1,
                    )
                    current_embeddings = self.vj_encoder.get_vision_features(
                        pixel_values_videos=current_videos
                    )
                    current_embeddings = torch.cat(
                        torch.chunk(current_embeddings, chunks=V, dim=0),
                        dim=2,
                    )
            #print(video_embeddings.shape) # [B, T//tubelet_size * dim_per_frame, V*embed_dim]
        
            # Step 3: VJ Predictor
            T = T // self.vj_encoder.config.tubelet_size
            if self.vj_predictor_variant == "f0_repeat_noncausal":
                input_states = current_embeddings
                gt_states = video_embeddings
            else:
                input_states = video_embeddings[:, :video_embeddings.shape[1] // T * (T-1),:]  # [B, (T-1)*dim_per_frame, V*embed_dim]
                gt_states = video_embeddings[:, video_embeddings.shape[1] // T:, :]
            #print(input_states.shape, action_tokens.shape)
            #exit()
            predicted_states = self.vj_predictor(
                input_states,
                action_tokens
            )

            teacher_forcing_wm_loss = F.l1_loss(
                predicted_states,
                gt_states,
                reduction="mean"
            )

            diagnostic_metrics = {}
            if compute_zero_code_metric:
                with torch.no_grad():
                    zero_code_predicted_states = self.vj_predictor(
                        input_states,
                        torch.zeros_like(action_tokens),
                    )
                    zero_code_wm_l1 = F.l1_loss(
                        zero_code_predicted_states,
                        gt_states,
                        reduction="mean",
                    )
                diagnostic_metrics = {
                    "normal_code_wm_l1_metric": teacher_forcing_wm_loss.detach(),
                    "zero_code_wm_l1_metric": zero_code_wm_l1,
                }
        
        if "action" not in examples[0]:
            return {"wm_loss": teacher_forcing_wm_loss, **diagnostic_metrics}

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = self.repeated_diffusion_steps
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                #print(state.shape)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            #print(embodied_action_repeated.shape, actions_target_repeated.shape, state_repeated.shape) if state_repeated is not None else print("No state for action model")
            #exit()
            action_loss = self.action_model(embodied_action_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)

        return {
            "action_loss": action_loss,
            "wm_loss": teacher_forcing_wm_loss * 0.1,
            **diagnostic_metrics,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt})
        
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        #embodied_action_indices = ~torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(embodied_action_tokens, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy()}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
