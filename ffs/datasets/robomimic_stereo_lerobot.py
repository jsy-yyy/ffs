from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from .lerobot_common import (
    ActionNormalizer,
    ROBOMIMIC_STEREO_LEROBOT,
    _feature_shape,
    _limit_rows_by_num_episodes,
    _normalize_episode_indices,
    _normalize_num_episodes,
    _resize_frame,
    _validate_episode_filter,
)


def _decode_image_bytes(image_bytes: bytes) -> torch.Tensor:
    with Image.open(BytesIO(image_bytes)) as image:
        array = np.asarray(image.convert("RGB"))
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float()


class RobomimicStereoLeRobotDataset(Dataset):
    """Robomimic LeRobot v3 stereo windows with image bytes stored in parquet."""

    action_mode = "raw"
    dataset_type = ROBOMIMIC_STEREO_LEROBOT

    def __init__(
        self,
        root: str | Path,
        camera_pairs: list[list[str]],
        num_history_frames: int,
        action_horizon: int,
        image_size: list[int] | tuple[int, int] | None = None,
        episode_indices: list[int] | tuple[int, ...] | set[int] | None = None,
        num_episodes: int | None = None,
        action_normalization: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.camera_pairs = camera_pairs
        self.num_history_frames = num_history_frames
        self.action_horizon = action_horizon
        self.image_size = tuple(image_size) if image_size is not None else None
        self.episode_indices = _normalize_episode_indices(episode_indices)
        self.num_episodes = _normalize_num_episodes(num_episodes)
        _validate_episode_filter(self.episode_indices, self.num_episodes)
        self._image_columns: dict[tuple[str, str], list[Any]] = {}

        with (self.root / "meta/info.json").open("r", encoding="utf-8") as f:
            self.info = json.load(f)
        self._validate_features()
        self.action_normalizer = ActionNormalizer(action_normalization, self.action_dim)

        self.rows = _limit_rows_by_num_episodes(self._load_rows(), self.num_episodes)
        self.samples = self._build_samples()

    def _validate_features(self) -> None:
        state_shape = _feature_shape(self.info, "observation.state")
        action_shape = _feature_shape(self.info, "action")
        if action_shape != [7]:
            raise ValueError(
                "robomimic-stereo-lerobot expects 7D robomimic delta actions. "
                f"Got action shape {action_shape}."
            )
        for left_key, right_key in self.camera_pairs:
            for image_key in (left_key, right_key):
                feature = self.info.get("features", {}).get(image_key)
                if feature is None:
                    raise ValueError(f"Dataset is missing required image feature {image_key!r}")
                if feature.get("dtype") != "image":
                    raise ValueError(f"Expected {image_key!r} to be an image feature, got {feature!r}")
        self.state_dim = int(state_shape[0])
        self.action_dim = int(action_shape[0])

    def _load_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        columns = ["observation.state", "action", "episode_index", "frame_index"]
        for data_file in sorted((self.root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(data_file, columns=columns).to_pydict()
            rel = data_file.relative_to(self.root).as_posix()
            count = len(table["episode_index"])
            for i in range(count):
                episode_index = int(table["episode_index"][i])
                if self.episode_indices is not None and episode_index not in self.episode_indices:
                    continue
                rows.append(
                    {
                        "state": table["observation.state"][i],
                        "action": table["action"][i],
                        "episode_index": episode_index,
                        "frame_index": table["frame_index"][i],
                        "data_file": rel,
                        "local_frame": i,
                    }
                )
        return rows

    def _build_samples(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        samples: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        groups: dict[tuple[str, int], list[int]] = {}
        for idx, row in enumerate(self.rows):
            groups.setdefault((row["data_file"], row["episode_index"]), []).append(idx)

        for indices in groups.values():
            for offset in range(len(indices)):
                history_start = offset - self.num_history_frames + 1
                pad_count = max(0, -history_start)
                history_offsets = [0] * pad_count + list(range(max(0, history_start), offset + 1))
                future_offsets = [min(i, len(indices) - 1) for i in range(offset, offset + self.action_horizon)]
                samples.append(
                    (
                        tuple(indices[i] for i in history_offsets),
                        tuple(indices[i] for i in future_offsets),
                    )
                )
        return samples

    def _image_column(self, image_key: str, data_file: str) -> list[Any]:
        cache_key = (image_key, data_file)
        column = self._image_columns.get(cache_key)
        if column is None:
            column = pq.read_table(self.root / data_file, columns=[image_key]).to_pydict()[image_key]
            self._image_columns[cache_key] = column
        return column

    def _read_frame(self, image_key: str, row_idx: int) -> torch.Tensor:
        row = self.rows[row_idx]
        value = self._image_column(image_key, row["data_file"])[row["local_frame"]]
        if isinstance(value, dict):
            image_bytes = value.get("bytes")
            if image_bytes is not None:
                return _resize_frame(_decode_image_bytes(image_bytes), self.image_size)
        raise ValueError(
            f"Expected image feature {image_key!r} in {row['data_file']} to contain PNG/JPEG bytes."
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_image_columns"] = {}
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        history, future = self.samples[idx]

        left_views = []
        right_views = []
        for left_key, right_key in self.camera_pairs:
            left_views.append(torch.stack([self._read_frame(left_key, i) for i in history]))
            right_views.append(torch.stack([self._read_frame(right_key, i) for i in history]))

        state = torch.tensor([self.rows[i]["state"] for i in history], dtype=torch.float32)
        raw_action = torch.tensor([self.rows[i]["action"] for i in future], dtype=torch.float32)
        action = self.action_normalizer.normalize_action(raw_action)

        return {
            "left": torch.stack(left_views, dim=1),
            "right": torch.stack(right_views, dim=1),
            "state": state,
            "action": action,
            "raw_action": raw_action,
        }

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_normalizer.normalize_action(action)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_normalizer.denormalize_action(action)


__all__ = ["RobomimicStereoLeRobotDataset"]
