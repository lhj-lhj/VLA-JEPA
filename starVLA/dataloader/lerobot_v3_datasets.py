"""Thin adapters from official LeRobot v3 datasets to starVLA examples.

LeRobot owns the v3 metadata, parquet/image decoding, episode boundary
clamping, delta timestamps, task lookup, and padding masks.  This file only
adapts decoded LIBERO or DROID samples to the list-of-dicts format consumed
by VLA_JEPA.
"""

from __future__ import annotations

import json
import logging
import os
import random
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from starVLA.posterior_view_sampling import (
    balanced_random_view_indices,
)


_LEROBOT_TIMESTAMP_TOLERANCE_ERROR = (
    "One or several query timestamps unexpectedly violate the tolerance"
)
_VIDEO_DECODE_ERROR_MARKERS = (
    "could not decode video",
    "failed to decode video",
    "failed to open video",
    "invalid data found when processing input",
    "moov atom not found",
    "no valid frames",
    "unable to decode video",
)


def _is_lerobot_video_decode_error(error: Exception) -> bool:
    """Recognize media failures without swallowing unrelated loader assertions."""

    message = str(error)
    if isinstance(error, AssertionError):
        return _LEROBOT_TIMESTAMP_TOLERANCE_ERROR in message
    if type(error).__module__.split(".", maxsplit=1)[0] == "av":
        return True
    lowered = message.lower()
    return any(marker in lowered for marker in _VIDEO_DECODE_ERROR_MARKERS)


def _video_path_from_decode_error(error: Exception) -> str | None:
    for line in str(error).splitlines():
        if line.startswith("video: "):
            return line.removeprefix("video: ").strip()
    return None


def collate_fn(
    batch: list[dict[str, Any]],
    *,
    posterior_view_sampling: str = "fixed",
    posterior_view_indices: tuple[int, int] = (0, 1),
) -> list[dict[str, Any]]:
    """Keep starVLA's existing list-of-example batch contract."""

    if posterior_view_sampling == "fixed":
        return batch
    if posterior_view_sampling != "balanced_random_single":
        raise ValueError(
            "posterior_view_sampling must be 'fixed' or "
            "'balanced_random_single'."
        )

    assignments = balanced_random_view_indices(
        batch_size=len(batch),
        view_indices=posterior_view_indices,
    )
    examples = []
    for sample, view_index in zip(batch, assignments):
        num_views = len(sample["video"])
        if not 0 <= view_index < num_views:
            raise ValueError(
                f"Posterior view {view_index} is out of range for {num_views} views."
            )
        example = dict(sample)
        example["posterior_video_view_index"] = view_index
        examples.append(example)
    return examples


def _load_lerobot_classes():
    """Import the official v3 reader lazily so other dataset modes stay usable."""

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    except ImportError as exc:
        raise ImportError(
            "LIBERO v3 loading requires the official LeRobot package. "
            "Install `lerobot==0.4.3` in the training environment."
        ) from exc

    try:
        installed_version = version("lerobot")
    except PackageNotFoundError:
        installed_version = "unknown"
    return LeRobotDataset, LeRobotDatasetMetadata, installed_version


def _as_vector(values: Any, *, name: str, expected_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected_dim,):
        raise ValueError(f"{name} must have shape ({expected_dim},), got {array.shape}.")
    return array


def _convert_libero_gripper_to_open(action: torch.Tensor) -> torch.Tensor:
    """Convert LIBERO env gripper {-1=open,+1=close} to starVLA {1=open,0=close}."""

    converted = action.clone()
    converted[..., -1] = (1.0 - converted[..., -1]) * 0.5
    return converted


def _convert_gripper_statistics(stats: dict[str, Any]) -> dict[str, Any]:
    """Apply y=(1-x)/2 to the last action dimension of LeRobot statistics."""

    converted = deepcopy(stats)
    vector_keys = ("mean", "std", "min", "max", "q01", "q99")
    old = {
        key: _as_vector(stats[key], name=f"action.{key}", expected_dim=7)
        for key in vector_keys
        if key in stats
    }

    for key in ("mean", "min", "max", "q01", "q99"):
        if key in converted:
            converted[key] = old[key].copy()
    if "std" in converted:
        converted["std"] = old["std"].copy()

    if "mean" in converted:
        converted["mean"][-1] = (1.0 - old["mean"][-1]) * 0.5
    if "std" in converted:
        converted["std"][-1] = old["std"][-1] * 0.5
    if "min" in converted and "max" in converted:
        converted["min"][-1] = (1.0 - old["max"][-1]) * 0.5
        converted["max"][-1] = (1.0 - old["min"][-1]) * 0.5
    if "q01" in converted and "q99" in converted:
        converted["q01"][-1] = (1.0 - old["q99"][-1]) * 0.5
        converted["q99"][-1] = (1.0 - old["q01"][-1]) * 0.5

    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in converted.items()
        if key in {"mean", "std", "min", "max", "q01", "q99"}
    }


def _convert_video_frames(
    frames: Any,
    *,
    horizon: int,
    current_size: int,
    video_size: int,
    key: str,
) -> tuple[Image.Image, np.ndarray]:
    """Resize official LeRobot CHW frames for the existing starVLA contract."""

    video = torch.as_tensor(frames, dtype=torch.float32)
    if video.ndim != 4 or tuple(video.shape[:2]) != (horizon, 3):
        raise ValueError(f"{key} must be [{horizon},3,H,W], got {tuple(video.shape)}.")
    video = F.interpolate(
        video,
        size=(video_size, video_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    video_uint8 = (
        video.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )
    current = Image.fromarray(video_uint8[0])
    if current.size != (current_size, current_size):
        current = current.resize(
            (current_size, current_size),
            resample=Image.Resampling.BILINEAR,
        )
    return current, video_uint8


class LeRobotV3VLADataset(Dataset):
    """Decode one combined LeRobot v3 LIBERO repository with official APIs."""

    def __init__(self, data_cfg: Any) -> None:
        LeRobotDataset, LeRobotDatasetMetadata, installed_version = _load_lerobot_classes()

        self.root = Path(data_cfg.data_root_dir)
        self.repo_id = str(data_cfg.get("repo_id", "HuggingFaceVLA/libero"))
        self.image_keys = list(data_cfg.img_keys)
        self.state_key = str(data_cfg.get("state_key", "observation.state"))
        self.action_key = str(data_cfg.get("action_key", "action"))
        self.task_key = str(data_cfg.get("task_key", "task"))
        self.action_horizon = int(data_cfg.action_horizon)
        self.video_horizon = int(data_cfg.video_horizon)
        self.resolution_size = int(data_cfg.get("resolution_size", 224))
        self.video_resolution_size = int(data_cfg.get("video_resolution_size", 256))
        self.with_state = bool(data_cfg.get("with_state", True))
        self.robot_tag = str(data_cfg.get("robot_tag", "franka"))

        if not self.root.is_dir():
            raise FileNotFoundError(f"LeRobot v3 root does not exist: {self.root}")
        if len(self.image_keys) != 2:
            raise ValueError(
                "The released VLA-JEPA checkpoints require exactly two camera views; "
                f"got {self.image_keys}."
            )

        self.meta = LeRobotDatasetMetadata(repo_id=self.repo_id, root=self.root)
        required_keys = set(self.image_keys + [self.action_key])
        if self.with_state:
            required_keys.add(self.state_key)
        missing = sorted(required_keys.difference(self.meta.features))
        if missing:
            raise KeyError(f"Missing required LeRobot v3 features: {missing}")
        if int(self.meta.fps) <= 0:
            raise ValueError(f"Dataset FPS must be positive, got {self.meta.fps}.")

        # This follows LeRobot's VLA-JEPA config exactly: t..t+7 observations
        # and t..t+6 actions. LeRobot handles episode-edge clamping and emits
        # `<feature>_is_pad` masks for out-of-range positions.
        delta_timestamps = {
            self.action_key: [step / self.meta.fps for step in range(self.action_horizon)]
        }
        for key in self.image_keys:
            delta_timestamps[key] = [step / self.meta.fps for step in range(self.video_horizon)]
        if self.with_state:
            delta_timestamps[self.state_key] = [
                step / self.meta.fps for step in range(self.video_horizon)
            ]

        self.dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.root,
            delta_timestamps=delta_timestamps,
            video_backend=str(data_cfg.get("video_backend", "pyav")),
        )
        self.lerobot_version = installed_version

        action_stats = self.meta.stats.get(self.action_key)
        if action_stats is None:
            raise KeyError(f"Missing action statistics for {self.action_key!r}.")
        self._action_min = torch.as_tensor(action_stats["min"], dtype=torch.float32)
        self._action_max = torch.as_tensor(action_stats["max"], dtype=torch.float32)
        if tuple(self._action_min.shape) != (7,) or tuple(self._action_max.shape) != (7,):
            raise ValueError(
                "VLA-JEPA LIBERO expects 7-D action statistics, got "
                f"{tuple(self._action_min.shape)} and {tuple(self._action_max.shape)}."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def _normalize_action(self, sample: dict[str, Any]) -> np.ndarray:
        action = torch.as_tensor(sample[self.action_key], dtype=torch.float32)
        if tuple(action.shape) != (self.action_horizon, 7):
            raise ValueError(
                f"Expected action [{self.action_horizon},7], got {tuple(action.shape)}."
            )

        action = _convert_libero_gripper_to_open(action)
        action_is_pad = sample.get(f"{self.action_key}_is_pad")
        if action_is_pad is not None:
            action[torch.as_tensor(action_is_pad, dtype=torch.bool)] = 0.0

        # Match the original starVLA LIBERO transform: min-max only the six
        # continuous dimensions; keep open_gripper in identity [0,1] space.
        low = self._action_min[:6]
        high = self._action_max[:6]
        scale = high - low
        if torch.any(scale == 0):
            raise ValueError("Continuous LIBERO action statistics contain a zero range.")
        action[:, :6] = 2.0 * (action[:, :6] - low) / scale - 1.0
        action[:, 6] = action[:, 6].clamp(0.0, 1.0)
        return action.numpy().astype(np.float32, copy=False)

    def _convert_video(self, frames: Any, *, key: str) -> tuple[Image.Image, np.ndarray]:
        return _convert_video_frames(
            frames,
            horizon=self.video_horizon,
            current_size=self.resolution_size,
            video_size=self.video_resolution_size,
            key=key,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        images: list[Image.Image] = []
        videos: list[np.ndarray] = []
        for key in self.image_keys:
            current, video = self._convert_video(sample[key], key=key)
            images.append(current)
            videos.append(video)

        example: dict[str, Any] = {
            "image": images,
            "video": np.stack(videos, axis=0),
            "lang": str(sample[self.task_key]),
            "action": self._normalize_action(sample),
            "video_fps": float(self.meta.fps),
        }
        if self.with_state:
            state = torch.as_tensor(sample[self.state_key], dtype=torch.float32)
            if state.ndim != 2 or state.shape[0] != self.video_horizon:
                raise ValueError(
                    f"{self.state_key} must be [{self.video_horizon},D], got {tuple(state.shape)}."
                )
            # Use the current state only. Future states are loaded solely because
            # LeRobot applies one observation window to all observation keys.
            example["state"] = state[:1].numpy().astype(np.float32, copy=False)
        return example

    @property
    def dataset_statistics(self) -> dict[str, Any]:
        """Return the checkpoint-side statistics format used by starVLA eval."""

        action_stats = _convert_gripper_statistics(self.meta.stats[self.action_key])
        action_stats["mask"] = [True, True, True, True, True, True, False]
        tag_stats: dict[str, Any] = {
            "action": action_stats,
            "num_transitions": int(self.meta.total_frames),
            "num_trajectories": int(self.meta.total_episodes),
        }
        if self.with_state:
            state_stats = self.meta.stats.get(self.state_key)
            if state_stats is None:
                raise KeyError(f"Missing state statistics for {self.state_key!r}.")
            tag_stats["state"] = {
                key: list(value)
                for key, value in state_stats.items()
                if key in {"mean", "std", "min", "max", "q01", "q99"}
            }
        return {self.robot_tag: tag_stats}


class LeRobotV3DROIDDataset(Dataset):
    """Thin starVLA adapter around official ``lerobot/droid_1.0.1`` decoding."""

    def __init__(self, data_cfg: Any) -> None:
        LeRobotDataset, LeRobotDatasetMetadata, installed_version = _load_lerobot_classes()

        self.root = Path(data_cfg.data_root_dir)
        self.repo_id = str(data_cfg.get("repo_id", "lerobot/droid_1.0.1"))
        self.image_keys = list(data_cfg.img_keys)
        self.world_model_image_keys = list(data_cfg.world_model_image_keys)
        self.continuous_action_key = str(
            data_cfg.get("continuous_action_key", "action.cartesian_velocity")
        )
        self.gripper_action_key = str(
            data_cfg.get("gripper_action_key", "action.gripper_position")
        )
        self.language_keys = list(
            data_cfg.get(
                "language_keys",
                [
                    "language_instruction",
                    "language_instruction_2",
                    "language_instruction_3",
                ],
            )
        )
        self.action_horizon = int(data_cfg.action_horizon)
        self.video_horizon = int(data_cfg.video_horizon)
        self.resolution_size = int(data_cfg.get("resolution_size", 224))
        self.video_resolution_size = int(data_cfg.get("video_resolution_size", 256))
        self.gripper_threshold = float(data_cfg.get("gripper_threshold", 0.5))
        self.robot_tag = str(data_cfg.get("robot_tag", "franka"))
        self.video_tolerance_s = float(data_cfg.get("video_tolerance_s", 1.0e-3))
        self.decode_max_retries = int(data_cfg.get("decode_max_retries", 8))
        self.decode_replacement_stride = int(
            data_cfg.get("decode_replacement_stride", 104729)
        )
        decode_error_log_dir = data_cfg.get("decode_error_log_dir")
        self.decode_error_log_dir = (
            Path(decode_error_log_dir) if decode_error_log_dir else None
        )
        episode_indices_path = Path(data_cfg.success_episode_indices_path)

        if not self.root.is_dir():
            raise FileNotFoundError(f"LeRobot v3 root does not exist: {self.root}")
        if not episode_indices_path.is_file():
            raise FileNotFoundError(
                f"DROID success episode manifest does not exist: {episode_indices_path}"
            )
        if bool(data_cfg.get("with_state", False)):
            raise ValueError("VLA-JEPA DROID pretraining requires with_state=false.")
        if len(self.world_model_image_keys) != 2:
            raise ValueError("VLA-JEPA requires exactly two world-model views.")
        if not set(self.world_model_image_keys).issubset(self.image_keys):
            raise ValueError("world_model_image_keys must be a subset of img_keys.")
        if self.video_tolerance_s <= 0:
            raise ValueError("video_tolerance_s must be positive.")
        if self.decode_max_retries < 0:
            raise ValueError("decode_max_retries must be non-negative.")
        if self.decode_replacement_stride <= 0:
            raise ValueError("decode_replacement_stride must be positive.")
        if self.decode_error_log_dir is not None:
            self.decode_error_log_dir.mkdir(parents=True, exist_ok=True)

        self.meta = LeRobotDatasetMetadata(repo_id=self.repo_id, root=self.root)
        action_keys = [self.continuous_action_key, self.gripper_action_key]
        required_keys = set(self.image_keys + action_keys + self.language_keys)
        missing = sorted(required_keys.difference(self.meta.features))
        if missing:
            raise KeyError(f"Missing required DROID v3 features: {missing}")

        manifest = json.loads(episode_indices_path.read_text(encoding="utf-8"))
        self.episode_indices = [int(index) for index in manifest["episode_indices"]]
        if not self.episode_indices or len(self.episode_indices) != len(
            set(self.episode_indices)
        ):
            raise ValueError("DROID success episode manifest is empty or has duplicates.")
        if min(self.episode_indices) < 0 or max(self.episode_indices) >= self.meta.total_episodes:
            raise ValueError("DROID success episode manifest contains invalid indices.")
        episode_index = np.asarray(self.meta.episodes["episode_index"], dtype=np.int64)
        episode_start = np.asarray(
            self.meta.episodes["dataset_from_index"], dtype=np.int64
        )
        episode_length = np.asarray(self.meta.episodes["length"], dtype=np.int64)
        start_by_episode = np.empty(self.meta.total_episodes, dtype=np.int64)
        length_by_episode = np.empty(self.meta.total_episodes, dtype=np.int64)
        start_by_episode[episode_index] = episode_start
        length_by_episode[episode_index] = episode_length
        self._selected_starts = start_by_episode[self.episode_indices]
        selected_lengths = length_by_episode[self.episode_indices]
        self._selected_ends = np.cumsum(selected_lengths, dtype=np.int64)
        self.num_selected_transitions = int(self._selected_ends[-1])
        self._quarantined_global_frame_indices = (
            self._load_quarantined_global_frame_indices()
        )

        # The official reader owns episode clamping, padding masks, and decoding.
        delta_timestamps = {
            key: [step / self.meta.fps for step in range(self.action_horizon)]
            for key in action_keys
        }
        delta_timestamps.update(
            {
                key: [step / self.meta.fps for step in range(self.video_horizon)]
                for key in self.image_keys
            }
        )

        self.dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.root,
            delta_timestamps=delta_timestamps,
            tolerance_s=self.video_tolerance_s,
            video_backend=str(data_cfg.get("video_backend", "pyav")),
        )
        self.lerobot_version = installed_version

        continuous_stats = self.meta.stats.get(self.continuous_action_key)
        gripper_stats = self.meta.stats.get(self.gripper_action_key)
        if continuous_stats is None or gripper_stats is None:
            raise KeyError("DROID v3 action statistics are incomplete.")
        self._continuous_stats = continuous_stats
        self._gripper_stats = gripper_stats
        self._continuous_min = torch.as_tensor(continuous_stats["min"], dtype=torch.float32)
        self._continuous_max = torch.as_tensor(continuous_stats["max"], dtype=torch.float32)

    def __len__(self) -> int:
        return self.num_selected_transitions

    def _sample_coordinates(self, index: int) -> tuple[int, int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = int(np.searchsorted(self._selected_ends, index, side="right"))
        previous_end = (
            0 if episode_position == 0 else int(self._selected_ends[episode_position - 1])
        )
        episode_frame_index = index - previous_end
        global_frame_index = int(
            self._selected_starts[episode_position] + episode_frame_index
        )
        return (
            global_frame_index,
            int(self.episode_indices[episode_position]),
            int(episode_frame_index),
        )

    def _global_frame_index(self, index: int) -> int:
        return self._sample_coordinates(index)[0]

    def _load_quarantined_global_frame_indices(self) -> set[int]:
        quarantined: set[int] = set()
        if self.decode_error_log_dir is None:
            return quarantined
        for log_path in sorted(self.decode_error_log_dir.glob("*.jsonl")):
            with log_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        record = json.loads(line)
                        if (
                            record.get("repo_id") == self.repo_id
                            and record.get("dataset_root") == str(self.root)
                        ):
                            quarantined.add(int(record["global_frame_index"]))
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        logging.warning(
                            "Ignoring malformed DROID decode quarantine record %s:%d",
                            log_path,
                            line_number,
                        )
        if quarantined:
            logging.info(
                "Loaded %d quarantined DROID global frame indices from %s",
                len(quarantined),
                self.decode_error_log_dir,
            )
        return quarantined

    def _record_decode_error(
        self,
        *,
        dataset_index: int,
        global_frame_index: int,
        episode_index: int,
        episode_frame_index: int,
        error: Exception,
    ) -> None:
        self._quarantined_global_frame_indices.add(global_frame_index)
        if self.decode_error_log_dir is None:
            return

        worker = get_worker_info()
        rank = os.environ.get("RANK", "0")
        worker_id = "main" if worker is None else str(worker.id)
        log_path = self.decode_error_log_dir / (
            f"rank-{rank}_worker-{worker_id}_pid-{os.getpid()}.jsonl"
        )
        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "repo_id": self.repo_id,
            "dataset_root": str(self.root),
            "dataset_index": dataset_index,
            "global_frame_index": global_frame_index,
            "episode_index": episode_index,
            "episode_frame_index": episode_frame_index,
            "video_path": _video_path_from_decode_error(error),
            "error_type": f"{type(error).__module__}.{type(error).__name__}",
            "error": str(error),
        }
        payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        file_descriptor = os.open(
            log_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            remaining = memoryview(payload)
            while remaining:
                remaining = remaining[os.write(file_descriptor, remaining) :]
        finally:
            os.close(file_descriptor)
        logging.warning(
            "Quarantined DROID dataset index %d (global=%d, episode=%d, frame=%d) "
            "after %s; a deterministic replacement will be used.",
            dataset_index,
            global_frame_index,
            episode_index,
            episode_frame_index,
            type(error).__name__,
        )

    def _decode_with_deterministic_replacement(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        last_decode_error: Exception | None = None
        for attempt in range(self.decode_max_retries + 1):
            candidate_index = (
                index + attempt * self.decode_replacement_stride
            ) % len(self)
            (
                global_frame_index,
                episode_index,
                episode_frame_index,
            ) = self._sample_coordinates(candidate_index)
            if global_frame_index in self._quarantined_global_frame_indices:
                continue
            try:
                return self.dataset[global_frame_index]
            except Exception as error:
                if not _is_lerobot_video_decode_error(error):
                    raise
                self._record_decode_error(
                    dataset_index=candidate_index,
                    global_frame_index=global_frame_index,
                    episode_index=episode_index,
                    episode_frame_index=episode_frame_index,
                    error=error,
                )
                last_decode_error = error

        message = (
            f"Unable to decode DROID dataset index {index} after "
            f"{self.decode_max_retries + 1} deterministic candidates."
        )
        if last_decode_error is not None:
            raise RuntimeError(message) from last_decode_error
        raise RuntimeError(
            message + " Every candidate was already present in the decode quarantine."
        )

    def _normalize_action(self, sample: dict[str, Any]) -> np.ndarray:
        continuous = torch.as_tensor(sample[self.continuous_action_key], dtype=torch.float32)
        gripper = torch.as_tensor(sample[self.gripper_action_key], dtype=torch.float32)
        if gripper.ndim == 1:
            gripper = gripper.unsqueeze(-1)
        if tuple(continuous.shape) != (self.action_horizon, 6) or tuple(
            gripper.shape
        ) != (self.action_horizon, 1):
            raise ValueError(
                "Expected DROID actions "
                f"[{self.action_horizon},6]+[{self.action_horizon},1], got "
                f"{tuple(continuous.shape)}+{tuple(gripper.shape)}."
            )

        scale = self._continuous_max - self._continuous_min
        if torch.any(scale == 0):
            raise ValueError("DROID Cartesian action statistics contain a zero range.")
        continuous = 2.0 * (continuous - self._continuous_min) / scale - 1.0
        # Public DROID stores larger gripper positions as open; starVLA uses
        # binary closedness, matching the released pretrain checkpoint.
        action = torch.cat(
            [continuous, (gripper <= self.gripper_threshold).float()], dim=-1
        )

        is_pad = torch.zeros(self.action_horizon, dtype=torch.bool)
        for key in (self.continuous_action_key, self.gripper_action_key):
            if f"{key}_is_pad" in sample:
                is_pad |= torch.as_tensor(sample[f"{key}_is_pad"], dtype=torch.bool).reshape(-1)
        action[is_pad] = 0.0
        return action.numpy().astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._decode_with_deterministic_replacement(index)
        language_candidates = [
            value
            for key in self.language_keys
            if (value := str(sample[key]).strip())
        ]
        if not language_candidates:
            raise RuntimeError(
                "Success-with-language manifest yielded an empty-language DROID sample."
            )
        # DROID provides up to three paraphrases per successful episode.
        # Resample one on every access so repeated frames/epochs augment language.
        language = random.choice(language_candidates)
        converted = {
            key: _convert_video_frames(
                sample[key],
                horizon=self.video_horizon,
                current_size=self.resolution_size,
                video_size=self.video_resolution_size,
                key=key,
            )
            for key in self.image_keys
        }

        return {
            "image": [converted[key][0] for key in self.image_keys],
            "video": np.stack(
                [converted[key][1] for key in self.world_model_image_keys],
                axis=0,
            ),
            "lang": language,
            "action": self._normalize_action(sample),
            "video_fps": float(self.meta.fps),
        }

    @property
    def dataset_statistics(self) -> dict[str, Any]:
        low = np.asarray(self._continuous_stats["min"], dtype=np.float64)
        high = np.asarray(self._continuous_stats["max"], dtype=np.float64)
        scale = high - low
        normalized = {
            key: (2.0 * (np.asarray(self._continuous_stats[key]) - low) / scale - 1.0)
            for key in ("mean", "q01", "q99")
        }
        normalized["std"] = 2.0 * np.asarray(self._continuous_stats["std"]) / scale
        normalized["min"], normalized["max"] = np.full(6, -1.0), np.full(6, 1.0)
        gripper = {
            "mean": 1.0 - float(self._gripper_stats["mean"][0]),
            "std": float(self._gripper_stats["std"][0]),
            "min": 0.0,
            "max": 1.0,
            "q01": 1.0 - float(self._gripper_stats["q99"][0]),
            "q99": 1.0 - float(self._gripper_stats["q01"][0]),
        }
        action_stats = {
            key: np.append(normalized[key], gripper[key]).tolist()
            for key in ("mean", "std", "min", "max", "q01", "q99")
        }
        action_stats["mask"] = [True] * 6 + [False]
        return {
            self.robot_tag: {
                "action": action_stats,
                "num_transitions": self.num_selected_transitions,
                "num_trajectories": len(self.episode_indices),
            }
        }


def get_lerobot_v3_datasets(data_cfg: Any) -> Dataset:
    profile = str(data_cfg.get("profile", "libero")).lower()
    if profile == "libero":
        return LeRobotV3VLADataset(data_cfg)
    if profile == "droid":
        return LeRobotV3DROIDDataset(data_cfg)
    raise ValueError(f"Unsupported LeRobot v3 profile: {profile!r}")
