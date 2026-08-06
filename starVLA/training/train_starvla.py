# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""
import warnings

# 全局忽略所有警告
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.run_root_dir, cfg.run_id, "tensorboard"))  # 保存目录

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            # self.vlm_train_dataloader
        )

        self.wandb_enabled = "wandb" in list(getattr(self.config, "trackers", []))
        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return

        wandb_dir = os.path.join(self.config.output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(
            name=self.config.run_id,
            dir=wandb_dir,
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            group="vla-train",
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self):
        """Save model weights and step metadata only."""

        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if self.accelerator.is_main_process:
                metrics = dict(metrics)
                learning_rates = self.lr_scheduler.get_last_lr()
                for group, lr in zip(self.optimizer.param_groups, learning_rates):
                    metrics[f"learning_rate/{group.get('name', 'unnamed')}"] = lr
                # Preserve the existing dashboard key for the first parameter group.
                metrics["learning_rate"] = learning_rates[0]

                # add epoch info
                metrics["epoch"] = round(
                    self.completed_steps
                    * self.accelerator.gradient_accumulation_steps
                    / len(self.vla_train_dataloader),
                    2,
                )

                # record to W&B
                if self.wandb_enabled:
                    wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Metrics: {metrics}")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    import torch

    def compare_state_dict(self, sd1, sd2, verbose=True):
        # 1. key 完全一致
        keys1 = set(sd1.keys())
        keys2 = set(sd2.keys())

        if keys1 != keys2:
            missing_1 = keys2 - keys1
            missing_2 = keys1 - keys2
            if verbose:
                if missing_1:
                    print("❌ sd1 缺少 keys:", missing_1)
                if missing_2:
                    print("❌ sd2 缺少 keys:", missing_2)
            return False

        # 2. 逐 tensor 比较
        for k in keys1:
            t1 = sd1[k]
            t2 = sd2[k]

            # 允许 Parameter
            if isinstance(t1, torch.nn.Parameter):
                t1 = t1.data
            if isinstance(t2, torch.nn.Parameter):
                t2 = t2.data

            # shape
            if t1.shape != t2.shape:
                if verbose:
                    print(f"❌ [{k}] shape 不一致: {t1.shape} vs {t2.shape}")
                return False

            # dtype
            if t1.dtype != t2.dtype:
                if verbose:
                    print(f"❌ [{k}] dtype 不一致: {t1.dtype} vs {t2.dtype}")
                return False

            # device 无所谓，统一搬到 CPU 比
            t1_cpu = t1.detach().cpu()
            t2_cpu = t2.detach().cpu()

            # 数值完全一致（bit 级）
            if not torch.equal(t1_cpu, t2_cpu):
                if verbose:
                    max_diff = (t1_cpu - t2_cpu).abs().max().item()
                    print(f"❌ [{k}] 数值不一致, max diff = {max_diff}")
                return False

        if verbose:
            print("✅ 两个 state_dict 完全一致")

        return True


    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        i = 0

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # Count/log/save only completed optimizer updates, not individual
            # gradient-accumulation microbatches.
            did_update = self.accelerator.sync_gradients
            if did_update:
                progress_bar.update(1)
                self.completed_steps += 1

                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

                # Evaluate, log, and save against the optimizer-step clock.
                eval_interval = int(getattr(self.config.trainer, "eval_interval", 0))
                if eval_interval > 0 and self.completed_steps % eval_interval == 0:
                    step_metrics = self.eval_action_model(step_metrics)

                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                step_metrics["step_time"] = t_end_model - t_start_data
                self._log_metrics(step_metrics)

                if self.completed_steps % self.config.trainer.save_interval == 0:
                    self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        if self.accelerator.is_main_process:

            examples = self._get_next_batch()

            score = 0.0
            num_samples = len(examples)

            batch_images = [example["image"] for example in examples]
            instructions = [example["lang"] for example in examples]  # [B, str]
            actions = [example["action"] for example in examples]  # label
            state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]


            # Predict actions using the model
            output_dict = self.model.predict_action(
                batch_images=batch_images, 
                instructions=instructions, 
                state=state,
                use_ddim=True,
                num_ddim_steps=20
            )

            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            mae_score = np.mean(np.abs(normalized_actions - actions))

            actions = np.array(actions)  # convert actions to numpy.ndarray
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score
            step_metrics["mae_score"] = mae_score
            self.writer.add_scalar("mae_score", step_metrics["mae_score"], self.completed_steps)
            self.writer.add_scalar("mse_score", step_metrics["mse_score"], self.completed_steps)
        
        pass
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            zero_code_metrics_frequency = int(
                getattr(self.config.trainer, "zero_code_metrics_frequency", 0)
            )
            forward_kwargs = {}
            if str(self.config.framework.name) == "VLA_JEPA":
                forward_kwargs["compute_zero_code_metric"] = (
                    self.accelerator.sync_gradients
                    and zero_code_metrics_frequency > 0
                    and (self.completed_steps + 1)
                    % zero_code_metrics_frequency
                    == 0
                )

            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla, **forward_kwargs)
                losses = {
                    key: value
                    for key, value in output_dict.items()
                    if key.endswith("_loss")
                }
                diagnostics = {
                    key: value
                    for key, value in output_dict.items()
                    if key.endswith("_metric")
                }
                if not losses:
                    raise RuntimeError("VLA batch did not return any '*_loss' tensors.")
                total_loss = sum(losses.values())

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # gradient clipping
            if (
                self.accelerator.sync_gradients
                and self.config.trainer.gradient_clipping is not None
            ):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            result_dict = {
                key: value.detach().float().item()
                for key, value in {**losses, **diagnostics}.items()
            }
            result_dict["total_loss"] = total_loss.detach().item()

        return result_dict

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        # close W&B
        if self.accelerator.is_main_process and self.wandb_enabled:
            wandb.finish()

        self.writer.close()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    accumulation_steps = int(
        cfg.trainer.get("gradient_accumulation_steps", 1)
    )
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            gradient_accumulation_steps=accumulation_steps,
        ),
        # ZeRO-2 cannot enter DeepSpeedEngine.no_sync(). Keep communication
        # enabled per microbatch and let DeepSpeed enforce the update boundary.
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=accumulation_steps,
            sync_each_batch=True,
        ),
    )
    accelerator.print(accelerator.state)
    logger.info("VLA Training :: Warming Up")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
