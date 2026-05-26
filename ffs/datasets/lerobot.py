from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import imageio
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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


class ActionNormalizer:
    """Optional quantile normalizer for selected action dimensions."""

    def __init__(self, cfg: dict[str, Any] | None, action_dim: int) -> None:
        self.enabled = False
        self.indices = torch.empty(0, dtype=torch.long)
        self.q01 = torch.empty(action_dim, dtype=torch.float32)
        self.q99 = torch.empty(action_dim, dtype=torch.float32)

        if cfg is None:
            return
        if not isinstance(cfg, dict):
            return
        if cfg.get("method") != "quantile":
            return

        normalize_indices = tuple(int(value) for value in cfg.get("normalize_indices", ()))
        stats = cfg.get("stats")
        if not isinstance(stats, dict):
            raise ValueError("dataset.action_normalization.stats must be provided.")
        q01 = stats.get("q01")
        q99 = stats.get("q99")
        if not isinstance(q01, list) or not isinstance(q99, list):
            raise ValueError("dataset.action_normalization.stats must contain q01 and q99 lists.")
        if len(q01) != action_dim or len(q99) != action_dim:
            raise ValueError(
                "dataset.action_normalization.stats q01/q99 must both have length "
                f"{action_dim}."
            )

        self.enabled = True
        self.indices = torch.tensor(normalize_indices, dtype=torch.long)
        self.q01 = torch.tensor(q01, dtype=torch.float32)
        self.q99 = torch.tensor(q99, dtype=torch.float32)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        out = action.clone()
        if not self.enabled:
            return out
        indices = self.indices.to(action.device)
        q01 = self.q01.to(device=action.device, dtype=action.dtype)[indices]
        q99 = self.q99.to(device=action.device, dtype=action.dtype)[indices]
        out[..., indices] = (out[..., indices] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return out

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        out = action.clone()
        if not self.enabled:
            return out
        indices = self.indices.to(action.device)
        q01 = self.q01.to(device=action.device, dtype=action.dtype)[indices]
        q99 = self.q99.to(device=action.device, dtype=action.dtype)[indices]
        out[..., indices] = (out[..., indices] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return out


def normalize_action(action: torch.Tensor, action_normalization_cfg: dict[str, Any] | None) -> torch.Tensor:
    return ActionNormalizer(action_normalization_cfg, action.shape[-1]).normalize_action(action)


def denormalize_action(action: torch.Tensor, action_normalization_cfg: dict[str, Any] | None) -> torch.Tensor:
    return ActionNormalizer(action_normalization_cfg, action.shape[-1]).denormalize_action(action)


class LeRobotStereoDataset(Dataset):
    """LeRobot v3 stereo windows for action chunk training."""

    def __init__(
        self,
        root: str | Path,
        camera_pairs: list[list[str]],
        num_history_frames: int,
        action_horizon: int,
        image_size: list[int] | tuple[int, int] | None = None,
        episode_indices: list[int] | tuple[int, ...] | set[int] | None = None,
        action_normalization: dict[str, Any] | None = None,
    ) -> None:
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "ffmpeg")
        self.root = Path(root)
        self.camera_pairs = camera_pairs
        self.num_history_frames = num_history_frames
        self.action_horizon = action_horizon
        self.image_size = tuple(image_size) if image_size is not None else None
        self.episode_indices = self._normalize_episode_indices(episode_indices)
        self._readers: dict[tuple[str, str], Any] = {}

        with (self.root / "meta/info.json").open("r", encoding="utf-8") as f:
            self.info = json.load(f)
        self._validate_features()
        self.action_normalizer = ActionNormalizer(action_normalization, self.action_dim)

        self.rows = self._load_rows()
        self.samples = self._build_samples()

    def _normalize_episode_indices(
        self,
        episode_indices: list[int] | tuple[int, ...] | set[int] | None,
    ) -> frozenset[int] | None:
        if episode_indices is None:
            return None
        normalized = frozenset(int(value) for value in episode_indices)
        if not normalized:
            raise ValueError("episode_indices must be non-empty when provided.")
        return normalized

    def _feature_shape(self, key: str) -> list[int]:
        try:
            shape = self.info["features"][key]["shape"]
        except KeyError as exc:
            raise ValueError(f"Dataset is missing required feature {key!r}") from exc
        if not isinstance(shape, list) or len(shape) != 1:
            raise ValueError(f"Expected {key!r} to be a 1D vector feature, got shape {shape!r}")
        return shape

    def _validate_features(self) -> None:
        state_shape = self._feature_shape("observation.state")
        action_shape = self._feature_shape("action")
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
            for offset in range(0, len(indices) - self.action_horizon + 1):
                history_start = offset - self.num_history_frames + 1
                pad_count = max(0, -history_start)
                history_offsets = [0] * pad_count + list(range(max(0, history_start), offset + 1))
                future_offsets = range(offset, offset + self.action_horizon)
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
        if self.image_size is not None:
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return frame

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
