import bisect
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_INSTRUCTION = "Completing something that humans might want to do."


def random_crop_or_pad(video, target_h, target_w, pad_value=0):
    """
    video: np.ndarray [T, H, W, 3]
    return: np.ndarray [T, target_h, target_w, 3]
    """
    _, height, width, channels = video.shape
    assert channels == 3

    top = random.randint(0, height - target_h) if height > target_h else 0
    left = random.randint(0, width - target_w) if width > target_w else 0

    cropped = video[
        :,
        top : top + min(height, target_h),
        left : left + min(width, target_w),
        :,
    ]
    output = np.full(
        (video.shape[0], target_h, target_w, 3),
        pad_value,
        dtype=video.dtype,
    )
    cropped_height, cropped_width = cropped.shape[1:3]
    output[:, :cropped_height, :cropped_width, :] = cropped
    return output


def resize_video(video, target_h, target_w):
    """
    video: np.ndarray [T, H, W, 3]
    return: np.ndarray [T, target_h, target_w, 3]
    """
    frames, _, _, channels = video.shape
    assert channels == 3

    output = np.empty((frames, target_h, target_w, 3), dtype=video.dtype)
    for frame_index in range(frames):
        output[frame_index] = np.asarray(
            Image.fromarray(video[frame_index]).resize(
                (target_w, target_h),
                resample=Image.Resampling.BILINEAR,
            )
        )
    return output


def _read_json_labels(path: Path, id2text: dict[int, str]) -> None:
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    for row in rows:
        if "id" not in row or "label" not in row:
            continue
        try:
            video_id = int(row["id"])
        except (TypeError, ValueError):
            continue
        id2text[video_id] = str(row["label"])


def _read_csv_labels(path: Path, id2text: dict[int, str]) -> None:
    rows = pd.read_csv(path, sep=";", header=None)
    for _, row in rows.iterrows():
        try:
            video_id = int(row.iloc[0])
        except (TypeError, ValueError):
            continue
        if len(row) > 1 and pd.notna(row.iloc[1]):
            id2text[video_id] = str(row.iloc[1])


def load_id2text(text_file: str) -> dict[int, str]:
    """Load SSV2 per-video text from the official JSON/CSV metadata."""
    text_path = Path(text_file)
    if not text_path.exists():
        raise FileNotFoundError(f"SSV2 metadata path does not exist: {text_path}")

    id2text: dict[int, str] = {}
    if text_path.is_dir():
        for name in ("train.json", "validation.json"):
            path = text_path / name
            if path.exists():
                _read_json_labels(path, id2text)
        test_answers = text_path / "test-answers.csv"
        if test_answers.exists():
            _read_csv_labels(test_answers, id2text)
    elif text_path.suffix.lower() == ".json":
        _read_json_labels(text_path, id2text)
    else:
        _read_csv_labels(text_path, id2text)

    if not id2text:
        raise RuntimeError(f"No SSV2 per-video labels found at {text_path}")
    return id2text


def collate_fn(batch, n_views=2, resolution_size=224):
    examples = []
    for video, instruction in batch:
        examples.append({
            "image": [Image.fromarray(video[0]).resize((resolution_size, resolution_size))],
            "video": np.stack([video.copy() for _ in range(n_views)], axis=0),
            "lang": instruction,
        })
    return examples


class VideoFolderDataset(Dataset):
    def __init__(
        self,
        video_dir: str,
        text_file: str,
        n_frames: int,
        extensions=(".mp4", ".avi", ".webm"),
        crop_h_size=420,
        crop_w_size=240,
        max_retry: int = 10,
    ):
        self.video_dir = video_dir
        self.n_frames = n_frames
        self.max_retry = max_retry
        self.crop_h_size = crop_h_size
        self.crop_w_size = crop_w_size
        self.extensions = tuple(extension.lower().lstrip(".") for extension in extensions)
        self.is_parquet = "parquet" in self.extensions
        self.id2text = load_id2text(text_file)
        self._cached_parquet_path = None
        self._cached_parquet_table = None

        if self.is_parquet:
            import pyarrow.parquet as pq

            self._pq = pq
            self.video_files = sorted(
                str(Path(video_dir) / filename)
                for filename in os.listdir(video_dir)
                if filename.lower().endswith(".parquet")
            )
            if not self.video_files:
                raise RuntimeError(f"No parquet files found in {video_dir}")
            row_counts = [pq.ParquetFile(path).metadata.num_rows for path in self.video_files]
            self._row_offsets = np.cumsum(row_counts).tolist()
            return

        suffixes = tuple(f".{extension}" for extension in self.extensions)
        self.video_files = sorted(filename for filename in os.listdir(video_dir) if filename.lower().endswith(suffixes))
        if not self.video_files:
            raise RuntimeError(f"No video files found in {video_dir}")

    def __len__(self):
        if self.is_parquet:
            return self._row_offsets[-1]
        return len(self.video_files)

    @staticmethod
    def _video_id_from_name(name):
        try:
            return int(Path(name).stem)
        except (TypeError, ValueError):
            return None

    def _decode_video_path(self, video_path):
        import av

        try:
            container = av.open(str(video_path))
        except av.error.FFmpegError as error:
            raise RuntimeError(f"Unable to open video: {video_path}") from error

        try:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        finally:
            container.close()

        if len(frames) < self.n_frames:
            raise ValueError(
                f"Video {video_path} has only {len(frames)} frames, "
                f"which is less than the required {self.n_frames} frames."
            )

        start = random.randint(0, len(frames) - self.n_frames)
        return resize_video(
            np.asarray(frames[start : start + self.n_frames]),
            target_h=self.crop_h_size,
            target_w=self.crop_w_size,
        )

    def _load_parquet_video(self, idx):
        shard_index = bisect.bisect_right(self._row_offsets, idx)
        previous_offset = 0 if shard_index == 0 else self._row_offsets[shard_index - 1]
        local_index = idx - previous_offset
        shard_path = self.video_files[shard_index]

        if self._cached_parquet_path != shard_path:
            self._cached_parquet_table = self._pq.read_table(shard_path, columns=["video"])
            self._cached_parquet_path = shard_path

        video_record = self._cached_parquet_table.column("video")[local_index].as_py()
        if not isinstance(video_record, dict) or video_record.get("bytes") is None:
            raise ValueError(f"Parquet row {idx} does not contain video bytes")

        video_name = video_record.get("path") or f"{idx}.webm"
        file_index = self._video_id_from_name(video_name)
        suffix = Path(video_name).suffix or ".webm"
        temporary_root = Path(
            os.environ.get(
                "VLA_JEPA_VIDEO_TMPDIR",
                str(Path(tempfile.gettempdir()) / "vla_jepa_video_decode"),
            )
        )
        temporary_root.mkdir(parents=True, exist_ok=True)

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, dir=temporary_root, delete=False) as file:
                file.write(video_record["bytes"])
                temporary_path = file.name
            frames = self._decode_video_path(temporary_path)
        finally:
            if temporary_path is not None:
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass

        instruction = self.id2text.get(file_index, DEFAULT_INSTRUCTION)
        return [frames, instruction]

    def _load_video(self, idx):
        if self.is_parquet:
            return self._load_parquet_video(idx)

        video_name = self.video_files[idx]
        file_index = self._video_id_from_name(video_name)
        video_path = os.path.join(self.video_dir, video_name)
        frames = self._decode_video_path(video_path)
        instruction = self.id2text.get(file_index, DEFAULT_INSTRUCTION)
        return [frames, instruction]

    def __getitem__(self, idx):
        last_error = None
        for _ in range(self.max_retry):
            try:
                return self._load_video(idx)
            except Exception as error:
                last_error = error
                idx = random.randint(0, len(self) - 1)

        raise RuntimeError(f"Unable to decode a video after {self.max_retry} attempts") from last_error
