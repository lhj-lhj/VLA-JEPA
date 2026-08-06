# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
#
# The spatial attention, RoPE, MLP, and initialization utilities reused here
# come from the official V-JEPA 2 action-conditioned predictor (MIT licensed).
# AdaLN conditioning and autoregressive rollout are local integration code.

"""Causal latent world model with per-layer AdaLN code conditioning."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from starVLA.model.modules.world_model.vj2_modules import (
    ACRoPEAttention,
    build_action_block_causal_attention_mask,
    DropPath,
    MLP,
    SwiGLUFFN,
)
from starVLA.model.modules.world_model.vj2_tensors import trunc_normal_


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply sample-wise or token-wise adaptive affine modulation."""

    if shift.ndim == x.ndim - 1:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    valid_shape = (
        shift.ndim == x.ndim
        and scale.ndim == x.ndim
        and shift.shape[0] == x.shape[0]
        and scale.shape[0] == x.shape[0]
        and shift.shape[-1] == x.shape[-1]
        and scale.shape[-1] == x.shape[-1]
        and shift.shape[1] in (1, x.shape[1])
        and scale.shape[1] in (1, x.shape[1])
    )
    if not valid_shape:
        raise ValueError(
            "AdaLN shift/scale must be [B,D] or match the hidden token shape."
        )
    return x * (1 + scale) + shift


def split_video_into_tubelets(
    videos: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    tubelet_size: int,
) -> torch.Tensor:
    """Split ``[B*V,T,C,H,W]`` videos into ordered isolated tubelet clips."""

    if videos.ndim != 5:
        raise ValueError(
            f"Expected videos [B*V,T,C,H,W], got {tuple(videos.shape)}."
        )
    if batch_size <= 0 or num_views <= 0 or tubelet_size <= 0:
        raise ValueError("Batch, view, and tubelet sizes must be positive.")
    if videos.shape[0] != batch_size * num_views:
        raise ValueError("Video batch does not match batch_size * num_views.")
    num_frames = videos.shape[1]
    if num_frames % tubelet_size:
        raise ValueError(
            f"{num_frames} frames cannot be divided into size-{tubelet_size} "
            "tubelets."
        )
    num_tubelets = num_frames // tubelet_size
    return videos.reshape(
        batch_size,
        num_views,
        num_tubelets,
        tubelet_size,
        *videos.shape[2:],
    ).flatten(0, 2)


class GaussianCodeProjector(nn.Module):
    """Shared diagonal-Gaussian head for posterior and prior Qwen codes."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        min_logvar: float = -6.0,
        max_logvar: float = 2.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive.")
        if min_logvar >= max_logvar:
            raise ValueError("min_logvar must be smaller than max_logvar.")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.norm = nn.LayerNorm(self.input_dim)
        self.to_stats = nn.Linear(self.input_dim, 2 * self.latent_dim)

        trunc_normal_(self.to_stats.weight, std=init_std)
        nn.init.zeros_(self.to_stats.bias)

    def forward(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if codes.ndim != 3 or codes.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected codes [B,N,{self.input_dim}], got {tuple(codes.shape)}."
            )
        mean, logvar = self.to_stats(self.norm(codes)).chunk(2, dim=-1)
        logvar = logvar.clamp(self.min_logvar, self.max_logvar)
        return mean, logvar

    @staticmethod
    def sample(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if deterministic:
            return mean
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    *,
    free_bits: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return free-bits KL objective and the unclamped mean KL.

    Computes KL(q_video || p_image+language) independently for every latent
    dimension. ``free_bits`` is expressed in nats per latent dimension.
    """

    if not (
        posterior_mean.shape
        == posterior_logvar.shape
        == prior_mean.shape
        == prior_logvar.shape
    ):
        raise ValueError("Posterior and prior Gaussian tensors must have identical shapes.")
    if free_bits < 0:
        raise ValueError("free_bits must be non-negative.")

    # KL is particularly sensitive to bf16 rounding in exp/log operations.
    q_mean = posterior_mean.float()
    q_logvar = posterior_logvar.float()
    p_mean = prior_mean.float()
    p_logvar = prior_logvar.float()
    kl_per_dim = 0.5 * (
        p_logvar
        - q_logvar
        + torch.exp(q_logvar - p_logvar)
        + (q_mean - p_mean).square() * torch.exp(-p_logvar)
        - 1.0
    )
    raw_kl = kl_per_dim.mean()
    if free_bits > 0:
        kl_per_dim = kl_per_dim.clamp_min(float(free_bits))
    return kl_per_dim.mean(), raw_kl


def balanced_diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    *,
    free_bits: float = 0.0,
    dynamics_scale: float = 1.0,
    representation_scale: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dreamer-style KL balancing for a diagonal Gaussian latent.

    The dynamics term updates only the image+language prior:

    ``KL(stop_gradient(q_video) || p_image+language)``.

    The representation term updates only the video posterior:

    ``KL(q_video || stop_gradient(p_image+language))``.

    Both terms have the same numerical KL value before free bits, but their
    gradient destinations differ. DreamerV3 uses the same stop-gradient split
    for its categorical RSSM posterior and prior.
    """

    if dynamics_scale < 0 or representation_scale < 0:
        raise ValueError("KL balance scales must be non-negative.")

    dynamics_kl, _ = diagonal_gaussian_kl(
        posterior_mean.detach(),
        posterior_logvar.detach(),
        prior_mean,
        prior_logvar,
        free_bits=free_bits,
    )
    representation_kl, raw_kl = diagonal_gaussian_kl(
        posterior_mean,
        posterior_logvar,
        prior_mean.detach(),
        prior_logvar.detach(),
        free_bits=free_bits,
    )
    balanced_kl = (
        float(dynamics_scale) * dynamics_kl
        + float(representation_scale) * representation_kl
    )
    return balanced_kl, dynamics_kl, representation_kl, raw_kl


class AdaLNWorldModelBlock(nn.Module):
    """V-JEPA 2 spatial block conditioned through AdaLN at both sublayers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        grid_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_silu: bool = False,
        wide_silu: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1.0e-6)
        self.attn = ACRoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_sdpa=True,
            is_causal=False,
            grid_size=grid_size,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1.0e-6)
        hidden_dim = int(dim * mlp_ratio)
        if use_silu:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=hidden_dim,
                act_layer=nn.SiLU,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_dim,
                act_layer=nn.GELU,
                drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # This predictor is intentionally random-initialized. Unlike adaLN-Zero,
        # a small nonzero initialization lets WM loss reach the code projector on
        # the first optimization step.
        trunc_normal_(self.adaLN_modulation[-1].weight, std=init_std)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        *,
        grid_height: int,
        grid_width: int,
        num_steps: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(condition).chunk(6, dim=-1)

        attn_input = _modulate(self.norm1(x), shift_attn, scale_attn)
        attn_output = self.attn(
            attn_input,
            mask=None,
            attn_mask=attention_mask,
            T=num_steps,
            H=grid_height,
            W=grid_width,
            action_tokens=0,
        )
        if gate_attn.ndim == x.ndim - 1:
            gate_attn = gate_attn.unsqueeze(1)
        x = x + gate_attn * self.drop_path(attn_output)

        mlp_input = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        if gate_mlp.ndim == x.ndim - 1:
            gate_mlp = gate_mlp.unsqueeze(1)
        x = x + gate_mlp * self.drop_path(self.mlp(mlp_input))
        return x


class CausalAdaLNWorldModel(nn.Module):
    """One-step latent predictor reused for a strictly causal rollout.

    Causality is structural: rollout step ``s`` receives only the previously
    predicted state and code group ``s``. Future code groups are never included
    in the same attention sequence.
    """

    def __init__(
        self,
        *,
        img_size: Tuple[int, int] = (256, 256),
        patch_size: int = 16,
        embed_dim: int = 2048,
        predictor_embed_dim: int = 1024,
        latent_dim: int = 256,
        code_tokens_per_step: int = 6,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_silu: bool = False,
        wide_silu: bool = True,
        use_activation_checkpointing: bool = False,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if len(img_size) != 2:
            raise ValueError("img_size must be a (height, width) pair.")
        if img_size[0] % patch_size or img_size[1] % patch_size:
            raise ValueError("img_size must be divisible by patch_size.")
        if predictor_embed_dim % num_heads:
            raise ValueError("predictor_embed_dim must be divisible by num_heads.")
        if code_tokens_per_step <= 0:
            raise ValueError("code_tokens_per_step must be positive.")

        self.embed_dim = int(embed_dim)
        self.predictor_embed_dim = int(predictor_embed_dim)
        self.latent_dim = int(latent_dim)
        self.code_tokens_per_step = int(code_tokens_per_step)
        self.grid_height = int(img_size[0] // patch_size)
        self.grid_width = int(img_size[1] // patch_size)
        self.tokens_per_frame = self.grid_height * self.grid_width
        self.use_activation_checkpointing = bool(use_activation_checkpointing)

        self.predictor_embed = nn.Linear(self.embed_dim, self.predictor_embed_dim)
        self.code_encoder = nn.Sequential(
            nn.LayerNorm(self.code_tokens_per_step * self.latent_dim),
            nn.Linear(
                self.code_tokens_per_step * self.latent_dim,
                self.predictor_embed_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.predictor_embed_dim, self.predictor_embed_dim),
        )

        drop_path_values = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                AdaLNWorldModelBlock(
                    self.predictor_embed_dim,
                    num_heads,
                    grid_size=self.grid_height,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_values[layer_index],
                    use_silu=use_silu,
                    wide_silu=wide_silu,
                    init_std=init_std,
                )
                for layer_index in range(depth)
            ]
        )
        self.predictor_norm = nn.LayerNorm(self.predictor_embed_dim)
        self.predictor_proj = nn.Linear(self.predictor_embed_dim, self.embed_dim)

        self.init_std = float(init_std)
        self.apply(self._init_weights)
        # ``apply`` also visits AdaLN linears, so restore their intended small
        # random initialization explicitly.
        for block in self.blocks:
            trunc_normal_(block.adaLN_modulation[-1].weight, std=self.init_std)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        self._rescale_blocks()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _rescale_blocks(self) -> None:
        def rescale(parameter: torch.Tensor, layer_id: int) -> None:
            parameter.div_(math.sqrt(2.0 * layer_id))

        for layer_id, block in enumerate(self.blocks, start=1):
            rescale(block.attn.proj.weight.data, layer_id)
            if hasattr(block.mlp, "fc2"):
                rescale(block.mlp.fc2.weight.data, layer_id)
            else:
                rescale(block.mlp.fc3.weight.data, layer_id)

    def _run_block(
        self,
        block: AdaLNWorldModelBlock,
        x: torch.Tensor,
        condition: torch.Tensor,
        *,
        num_steps: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return block(
            x,
            condition,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_steps=num_steps,
            attention_mask=attention_mask,
        )

    def predict_sequence(
        self,
        states: torch.Tensor,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a state sequence from aligned context states and codes.

        During training, ``states`` are shifted ground-truth tubelets
        (teacher forcing). Official V-JEPA2 block-causal attention lets each
        time step attend only to its current and previous context states.
        """

        if states.ndim != 4 or states.shape[2:] != (
            self.tokens_per_frame,
            self.embed_dim,
        ):
            raise ValueError(
                "Expected states "
                f"[B,S,{self.tokens_per_frame},{self.embed_dim}], "
                f"got {tuple(states.shape)}."
            )
        if codes.ndim != 4 or codes.shape[2:] != (
            self.code_tokens_per_step,
            self.latent_dim,
        ):
            raise ValueError(
                "Expected codes "
                f"[B,S,{self.code_tokens_per_step},{self.latent_dim}], "
                f"got {tuple(codes.shape)}."
            )
        if states.shape[:2] != codes.shape[:2]:
            raise ValueError("State and code batch/time dimensions must match.")

        batch_size, num_steps = states.shape[:2]
        x = self.predictor_embed(states).flatten(1, 2)
        condition = self.code_encoder(codes.flatten(2))
        condition = (
            condition.unsqueeze(2)
            .expand(-1, -1, self.tokens_per_frame, -1)
            .flatten(1, 2)
        )
        attention_mask = build_action_block_causal_attention_mask(
            num_steps,
            self.grid_height,
            self.grid_width,
            add_tokens=0,
        ).to(x.device, non_blocking=True)

        for block in self.blocks:
            if self.use_activation_checkpointing and self.training:
                x = checkpoint(
                    lambda hidden, cond, current_block=block: self._run_block(
                        current_block,
                        hidden,
                        cond,
                        num_steps=num_steps,
                        attention_mask=attention_mask,
                    ),
                    x,
                    condition,
                    use_reentrant=False,
                )
            else:
                x = self._run_block(
                    block,
                    x,
                    condition,
                    num_steps=num_steps,
                    attention_mask=attention_mask,
                )
        x = self.predictor_proj(self.predictor_norm(x))
        return x.view(
            batch_size,
            num_steps,
            self.tokens_per_frame,
            self.embed_dim,
        )

    def teacher_forced(
        self,
        ground_truth_contexts: torch.Tensor,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """Predict all next states from shifted ground-truth contexts."""

        return self.predict_sequence(ground_truth_contexts, codes)

    def forward(self, state: torch.Tensor, code_group: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError("One-step state must be [B,P,D].")
        if code_group.ndim != 3:
            raise ValueError("One-step code group must be [B,K,D].")
        return self.predict_sequence(
            state.unsqueeze(1),
            code_group.unsqueeze(1),
        ).squeeze(1)

    def rollout(self, initial_state: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Autoregressively predict one state per code group.

        Args:
            initial_state: ``[B, patches, embed_dim]``.
            codes: ``[B, steps, code_tokens_per_step, latent_dim]``.

        Returns:
            Predicted states ``[B, steps, patches, embed_dim]``.
        """

        if codes.ndim != 4 or codes.shape[2:] != (
            self.code_tokens_per_step,
            self.latent_dim,
        ):
            raise ValueError(
                "Expected rollout codes "
                f"[B,S,{self.code_tokens_per_step},{self.latent_dim}], "
                f"got {tuple(codes.shape)}."
            )
        if initial_state.shape[0] != codes.shape[0]:
            raise ValueError("Initial-state and code batch sizes must match.")

        context_states = [initial_state]
        predictions = []
        for step_index in range(codes.shape[1]):
            available_states = torch.stack(context_states, dim=1)
            available_codes = codes[:, : step_index + 1]
            next_state = self.predict_sequence(
                available_states,
                available_codes,
            )[:, -1]
            predictions.append(next_state)
            context_states.append(next_state)
        return torch.stack(predictions, dim=1)
