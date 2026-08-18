from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import imageio
import torch
from torch.utils.data import Dataset

from .lerobot_common import (
    ROBOTWIN_HDF5_DATASET,
    _normalize_episode_indices,
    _normalize_num_episodes,
    _resize_frame,
    _validate_episode_filter,
)

DEFAULT_ROBOTWIN_HDF5_ROOT = "/mnt/nas/datasets5/jsy_robotwin"
EPISODE_RE = re.compile(r"episode(\d+)\.hdf5$")


@dataclass(frozen=True)
class EpisodeRecord:
    task: str
    config: str
    episode_index: int
    hdf5_path: Path
    video_dir: Path
    length: int


class RobotwinHdf5Dataset(Dataset):
    """Native RoboTwin HDF5 stereo windows with absolute EEF action targets."""

    action_mode = "absolute-eef"
    dataset_type = ROBOTWIN_HDF5_DATASET
    state_dim = 16
    action_dim = 16

    def __init__(
        self,
        root: str | Path = DEFAULT_ROBOTWIN_HDF5_ROOT,
        camera_pairs: list[list[str]] | tuple[tuple[str, str], ...] | None = None,
        num_history_frames: int = 1,
        action_horizon: int = 1,
        image_size: list[int] | tuple[int, int] | None = None,
        episode_indices: list[int] | tuple[int, ...] | set[int] | None = None,
        num_episodes: int | None = None,
        tasks: list[str] | tuple[str, ...] | None = None,
        configs: list[str] | tuple[str, ...] | None = None,
        skip_incomplete: bool = True,
        max_scan_workers: int = 16,
        max_open_video_readers: int = 32,
        action_normalization: dict[str, Any] | None = None,
    ) -> None:
        del action_normalization
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "ffmpeg")
        self.root = Path(root)
        self.camera_pairs = [list(pair) for pair in (camera_pairs or [["head_stereo.left", "head_stereo.right"]])]
        self.num_history_frames = int(num_history_frames)
        self.action_horizon = int(action_horizon)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.episode_indices = _normalize_episode_indices(episode_indices)
        self.num_episodes = _normalize_num_episodes(num_episodes)
        _validate_episode_filter(self.episode_indices, self.num_episodes)
        self.tasks = tuple(str(task) for task in tasks) if tasks is not None else None
        self.configs = tuple(str(config) for config in configs) if configs is not None else None
        self.skip_incomplete = bool(skip_incomplete)
        self.max_scan_workers = max(1, int(max_scan_workers))
        self.max_open_video_readers = max(1, int(max_open_video_readers))
        self._readers: OrderedDict[Path, Any] = OrderedDict()
        self._state_cache: dict[int, torch.Tensor] = {}

        if self.num_history_frames <= 0:
            raise ValueError("num_history_frames must be positive.")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")

        self.episodes = self._discover_episodes()
        self.id_to_task = self._build_task_vocab()
        self.task_to_id = {task: idx for idx, task in enumerate(self.id_to_task)}
        self.num_tasks = len(self.id_to_task)
        self.samples = self._build_samples()

    def _discover_episodes(self) -> list[EpisodeRecord]:
        candidates: list[EpisodeRecord] = []
        for task_dir in self._task_dirs():
            for config_dir in self._config_dirs(task_dir):
                data_dir = config_dir / "data"
                if not data_dir.is_dir():
                    continue
                for hdf5_path in sorted(data_dir.glob("episode*.hdf5"), key=self._episode_sort_key):
                    episode_index = self._episode_index(hdf5_path)
                    if self.episode_indices is not None and episode_index not in self.episode_indices:
                        continue
                    record = EpisodeRecord(
                        task=task_dir.name,
                        config=config_dir.name,
                        episode_index=episode_index,
                        hdf5_path=hdf5_path,
                        video_dir=config_dir / "video" / f"episode{episode_index}",
                        length=0,
                    )
                    if self._has_required_videos(record):
                        candidates.append(record)
                    elif not self.skip_incomplete:
                        missing = ", ".join(str(path) for path in self._required_video_paths(record) if not path.is_file())
                        raise FileNotFoundError(f"Missing RoboTwin video files for {hdf5_path}: {missing}")

        if self.num_episodes is not None:
            candidates = candidates[: self.num_episodes]
        if not candidates:
            raise ValueError(
                "No valid RoboTwin HDF5 episodes found under "
                f"{self.root} for tasks={self.tasks} configs={self.configs}."
            )

        with ThreadPoolExecutor(max_workers=self.max_scan_workers) as executor:
            lengths = list(executor.map(lambda record: self._episode_length(record.hdf5_path), candidates))
        return [
            EpisodeRecord(
                task=record.task,
                config=record.config,
                episode_index=record.episode_index,
                hdf5_path=record.hdf5_path,
                video_dir=record.video_dir,
                length=length,
            )
            for record, length in zip(candidates, lengths)
        ]

    def _task_dirs(self) -> list[Path]:
        if self.tasks is not None:
            return [self.root / task for task in self.tasks]
        return sorted(path for path in self.root.iterdir() if path.is_dir())

    def _build_task_vocab(self) -> tuple[str, ...]:
        if self.tasks is not None:
            return tuple(self.tasks)
        ordered = []
        seen = set()
        for record in self.episodes:
            if record.task not in seen:
                seen.add(record.task)
                ordered.append(record.task)
        return tuple(ordered)

    def _config_dirs(self, task_dir: Path) -> list[Path]:
        if not task_dir.is_dir():
            return []
        if self.configs is not None:
            return [task_dir / config for config in self.configs]
        return sorted(path for path in task_dir.iterdir() if path.is_dir())

    @staticmethod
    def _episode_sort_key(path: Path) -> tuple[int, str]:
        match = EPISODE_RE.search(path.name)
        if match is None:
            return (10**12, path.name)
        return (int(match.group(1)), path.name)

    @staticmethod
    def _episode_index(path: Path) -> int:
        match = EPISODE_RE.search(path.name)
        if match is None:
            raise ValueError(f"Expected RoboTwin episode filename like episode0.hdf5, got {path.name!r}.")
        return int(match.group(1))

    @staticmethod
    def _episode_length(path: Path) -> int:
        with h5py.File(path, "r") as f:
            length = int(f["endpose/left_endpose"].shape[0])
            for key in (
                "endpose/left_gripper",
                "endpose/right_endpose",
                "endpose/right_gripper",
            ):
                if int(f[key].shape[0]) != length:
                    raise ValueError(f"RoboTwin episode {path} has inconsistent length for {key}.")
        return length

    def _required_video_paths(self, record: EpisodeRecord) -> set[Path]:
        paths = set()
        for pair in self.camera_pairs:
            if len(pair) != 2:
                raise ValueError(f"Each camera pair must contain two keys, got {pair!r}.")
            for key in pair:
                paths.add(self._video_path(record, key))
        return paths

    def _has_required_videos(self, record: EpisodeRecord) -> bool:
        return all(path.is_file() for path in self._required_video_paths(record))

    @staticmethod
    def _video_path(record: EpisodeRecord, key: str) -> Path:
        if key in {"head_stereo.left", "head_stereo.right"}:
            return record.video_dir / "head_stereo.mp4"
        if "." in key:
            raise ValueError(
                "RoboTwin HDF5 camera keys must be head_stereo.left, head_stereo.right, "
                f"or a direct video stem such as left/right. Got {key!r}."
            )
        return record.video_dir / f"{key}.mp4"

    def _build_samples(self) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
        samples: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
        for episode_idx, record in enumerate(self.episodes):
            for offset in range(record.length):
                history_start = offset - self.num_history_frames + 1
                pad_count = max(0, -history_start)
                history_offsets = [0] * pad_count + list(range(max(0, history_start), offset + 1))
                future_offsets = [min(i, record.length - 1) for i in range(offset, offset + self.action_horizon)]
                samples.append((episode_idx, tuple(history_offsets), tuple(future_offsets)))
        return samples

    @staticmethod
    def _read_episode_state(path: Path) -> torch.Tensor:
        with h5py.File(path, "r") as f:
            left_endpose = torch.from_numpy(f["endpose/left_endpose"][:]).float()
            left_gripper = torch.from_numpy(f["endpose/left_gripper"][:]).float().unsqueeze(-1)
            right_endpose = torch.from_numpy(f["endpose/right_endpose"][:]).float()
            right_gripper = torch.from_numpy(f["endpose/right_gripper"][:]).float().unsqueeze(-1)
        return torch.cat((left_endpose, left_gripper, right_endpose, right_gripper), dim=-1)

    def _episode_state(self, episode_idx: int) -> torch.Tensor:
        state = self._state_cache.get(episode_idx)
        if state is None:
            state = self._read_episode_state(self.episodes[episode_idx].hdf5_path)
            self._state_cache[episode_idx] = state
        return state

    def _reader(self, path: Path) -> Any:
        reader = self._readers.get(path)
        if reader is not None:
            self._readers.move_to_end(path)
            return reader
        reader = imageio.get_reader(path, "ffmpeg")
        self._readers[path] = reader
        self._evict_video_readers()
        return reader

    @staticmethod
    def _close_reader(reader: Any) -> None:
        close = getattr(reader, "close", None)
        if callable(close):
            close()

    def _evict_video_readers(self) -> None:
        while len(self._readers) > self.max_open_video_readers:
            _, reader = self._readers.popitem(last=False)
            self._close_reader(reader)

    def close(self) -> None:
        for reader in self._readers.values():
            self._close_reader(reader)
        self._readers.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _read_frame(self, record: EpisodeRecord, key: str, frame_index: int) -> torch.Tensor:
        path = self._video_path(record, key)
        frame = self._reader(path).get_data(frame_index)
        if key == "head_stereo.left":
            frame = frame[:, : frame.shape[1] // 2]
        elif key == "head_stereo.right":
            frame = frame[:, frame.shape[1] // 2 :]
        frame_tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float()
        return _resize_frame(frame_tensor, self.image_size)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = {}
        state["_state_cache"] = {}
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_idx, history, future = self.samples[idx]
        record = self.episodes[episode_idx]

        left_views = []
        right_views = []
        for left_key, right_key in self.camera_pairs:
            left_views.append(torch.stack([self._read_frame(record, left_key, i) for i in history]))
            right_views.append(torch.stack([self._read_frame(record, right_key, i) for i in history]))

        episode_state = self._episode_state(episode_idx)
        state = episode_state[list(history)]
        absolute_action = episode_state[list(future)]
        action = absolute_action.clone()

        return {
            "left": torch.stack(left_views, dim=1),
            "right": torch.stack(right_views, dim=1),
            "state": state,
            "action": action,
            "absolute_action": absolute_action,
            "task_id": torch.tensor(self.task_to_id[record.task], dtype=torch.long),
            "task_name": record.task,
        }

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.clone()

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.clone()


__all__ = ["DEFAULT_ROBOTWIN_HDF5_ROOT", "RobotwinHdf5Dataset"]
