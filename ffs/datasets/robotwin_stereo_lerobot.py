from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import imageio
import pyarrow.parquet as pq
import torch
from torch.utils.data import ConcatDataset, Dataset

from .lerobot_common import (
    ActionNormalizer,
    ROBOTWIN_STEREO_LEROBOT,
    _feature_shape,
    _limit_rows_by_num_episodes,
    _normalize_episode_indices,
    _normalize_num_episodes,
    _resize_frame,
    _validate_episode_filter,
)


DUAL_ARM_EEF_NAMES = [
    "left_endpose_x",
    "left_endpose_y",
    "left_endpose_z",
    "left_endpose_qw",
    "left_endpose_qx",
    "left_endpose_qy",
    "left_endpose_qz",
    "left_gripper",
    "right_endpose_x",
    "right_endpose_y",
    "right_endpose_z",
    "right_endpose_qw",
    "right_endpose_qx",
    "right_endpose_qy",
    "right_endpose_qz",
    "right_gripper",
]

POSITION_ACTION_INDICES = (0, 1, 2, 8, 9, 10)


def _normalize_quaternion(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def _canonicalize_quaternion(quat: torch.Tensor) -> torch.Tensor:
    sign = torch.where(quat[..., :1] < 0, -1.0, 1.0)
    return quat * sign


def _quaternion_conjugate(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quaternion_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _rotate_vector_by_inverse_quaternion(vector: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quaternion(quat)
    inv_quat = _quaternion_conjugate(quat)
    vector_quat = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quaternion_multiply(_quaternion_multiply(inv_quat, vector_quat), quat)[..., 1:]


def _rotate_vector_by_quaternion(vector: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quaternion(quat)
    vector_quat = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quaternion_multiply(
        _quaternion_multiply(quat, vector_quat),
        _quaternion_conjugate(quat),
    )[..., 1:]


def absolute_action_to_relative_eef_pose(action: torch.Tensor, base_state: torch.Tensor) -> torch.Tensor:
    """Convert absolute dual-arm EEF action targets to poses relative to base_state."""
    relative = action.clone()
    base = base_state.unsqueeze(-2) if base_state.ndim == action.ndim - 1 else base_state
    for start in (0, 8):
        target_pos = action[..., start : start + 3]
        target_quat = _normalize_quaternion(action[..., start + 3 : start + 7])
        base_pos = base[..., start : start + 3]
        base_quat = _normalize_quaternion(base[..., start + 3 : start + 7])

        relative[..., start : start + 3] = _rotate_vector_by_inverse_quaternion(
            target_pos - base_pos,
            base_quat,
        )
        base_inv = _quaternion_conjugate(base_quat)
        rel_quat = _quaternion_multiply(base_inv, target_quat)
        relative[..., start + 3 : start + 7] = _canonicalize_quaternion(
            _normalize_quaternion(rel_quat)
        )
    return relative


def relative_action_to_absolute_eef_pose(action: torch.Tensor, base_state: torch.Tensor) -> torch.Tensor:
    """Convert relative dual-arm EEF action targets back to absolute poses."""
    absolute = action.clone()
    base = base_state.unsqueeze(-2) if base_state.ndim == action.ndim - 1 else base_state
    for start in (0, 8):
        rel_pos = action[..., start : start + 3]
        rel_quat = _normalize_quaternion(action[..., start + 3 : start + 7])
        base_pos = base[..., start : start + 3]
        base_quat = _normalize_quaternion(base[..., start + 3 : start + 7])

        absolute[..., start : start + 3] = _rotate_vector_by_quaternion(rel_pos, base_quat) + base_pos
        abs_quat = _quaternion_multiply(base_quat, rel_quat)
        absolute[..., start + 3 : start + 7] = _canonicalize_quaternion(
            _normalize_quaternion(abs_quat)
        )
    return absolute


class RobotwinStereoLeRobotDataset(Dataset):
    """RoboTwin LeRobot v3 stereo windows for action chunk training."""

    action_mode = "relative-eef"
    dataset_type = ROBOTWIN_STEREO_LEROBOT

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
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "ffmpeg")
        self.root = Path(root)
        self.camera_pairs = camera_pairs
        self.num_history_frames = num_history_frames
        self.action_horizon = action_horizon
        self.image_size = tuple(image_size) if image_size is not None else None
        self.episode_indices = _normalize_episode_indices(episode_indices)
        self.num_episodes = _normalize_num_episodes(num_episodes)
        _validate_episode_filter(self.episode_indices, self.num_episodes)
        self._readers: dict[tuple[str, str], Any] = {}

        with (self.root / "meta/info.json").open("r", encoding="utf-8") as f:
            self.info = json.load(f)
        self._validate_features()
        self.action_normalizer = ActionNormalizer(action_normalization, self.action_dim)

        self.rows = _limit_rows_by_num_episodes(self._load_rows(), self.num_episodes)
        self.samples = self._build_samples()

    def _validate_features(self) -> None:
        state_shape = _feature_shape(self.info, "observation.state")
        action_shape = _feature_shape(self.info, "action")
        state_names = self.info["features"]["observation.state"].get("names")
        action_names = self.info["features"]["action"].get("names")
        if state_shape != action_shape or state_names != action_names:
            raise ValueError(
                "This loader expects the new dataset format where action is the "
                "next-step EEF pose with the same shape/names as observation.state. "
                f"Got state shape/names {state_shape}/{state_names} and "
                f"action shape/names {action_shape}/{action_names}."
            )
        if state_names != DUAL_ARM_EEF_NAMES:
            raise ValueError(
                "This loader expects 16D dual-arm EEF pose features ordered as "
                f"{DUAL_ARM_EEF_NAMES}. Got {state_names}."
            )
        self.state_dim = state_shape[0]
        self.action_dim = action_shape[0]

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

    def _video_path(self, video_key: str, data_file: str) -> Path:
        rel = data_file.replace("data/", f"videos/{video_key}/").replace(".parquet", ".mp4")
        return self.root / rel

    def _reader(self, video_key: str, data_file: str) -> Any:
        cache_key = (video_key, data_file)
        reader = self._readers.get(cache_key)
        if reader is None:
            reader = imageio.get_reader(self._video_path(video_key, data_file), "ffmpeg")
            self._readers[cache_key] = reader
        return reader

    def _read_frame(self, video_key: str, row_idx: int) -> torch.Tensor:
        row = self.rows[row_idx]
        frame = self._reader(video_key, row["data_file"]).get_data(row["local_frame"])
        frame = torch.from_numpy(frame.copy()).permute(2, 0, 1).float()
        return _resize_frame(frame, self.image_size)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = {}
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
        absolute_action = torch.tensor([self.rows[i]["action"] for i in future], dtype=torch.float32)
        relative_action = absolute_action_to_relative_eef_pose(absolute_action, state[-1])
        action = self.action_normalizer.normalize_action(relative_action)

        return {
            "left": torch.stack(left_views, dim=1),
            "right": torch.stack(right_views, dim=1),
            "state": state,
            "action": action,
            "relative_action": relative_action,
            "absolute_action": absolute_action,
        }

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_normalizer.normalize_action(action)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_normalizer.denormalize_action(action)


class MultiRobotwinStereoLeRobotDataset(Dataset):
    """Concatenate compatible RoboTwin LeRobot stereo datasets without copying data."""

    action_mode = RobotwinStereoLeRobotDataset.action_mode
    dataset_type = ROBOTWIN_STEREO_LEROBOT

    def __init__(
        self,
        roots: list[str | Path] | tuple[str | Path, ...],
        camera_pairs: list[list[str]],
        num_history_frames: int,
        action_horizon: int,
        image_size: list[int] | tuple[int, int] | None = None,
        episode_indices: list[int] | tuple[int, ...] | set[int] | None = None,
        num_episodes: int | None = None,
        action_normalization: dict[str, Any] | None = None,
    ) -> None:
        if not roots:
            raise ValueError("dataset.roots must be non-empty when provided.")

        self.datasets = [
            RobotwinStereoLeRobotDataset(
                root=root,
                camera_pairs=camera_pairs,
                num_history_frames=num_history_frames,
                action_horizon=action_horizon,
                image_size=image_size,
                episode_indices=episode_indices,
                num_episodes=num_episodes,
                action_normalization=action_normalization,
            )
            for root in roots
        ]
        first = self.datasets[0]
        self.roots = [dataset.root for dataset in self.datasets]
        self.state_dim = first.state_dim
        self.action_dim = first.action_dim
        self.action_normalizer = first.action_normalizer
        self.sample_counts = [len(dataset) for dataset in self.datasets]

        for dataset in self.datasets[1:]:
            if dataset.state_dim != self.state_dim:
                raise ValueError(
                    f"All datasets must have state_dim={self.state_dim}; "
                    f"{dataset.root} has state_dim={dataset.state_dim}."
                )
            if dataset.action_dim != self.action_dim:
                raise ValueError(
                    f"All datasets must have action_dim={self.action_dim}; "
                    f"{dataset.root} has action_dim={dataset.action_dim}."
                )

        self._concat = ConcatDataset(self.datasets)

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._concat[idx]

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_normalizer.normalize_action(action)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_normalizer.denormalize_action(action)


__all__ = [
    "DUAL_ARM_EEF_NAMES",
    "MultiRobotwinStereoLeRobotDataset",
    "POSITION_ACTION_INDICES",
    "RobotwinStereoLeRobotDataset",
    "absolute_action_to_relative_eef_pose",
    "relative_action_to_absolute_eef_pose",
]
