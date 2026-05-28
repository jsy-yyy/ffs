from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from .lerobot_common import (
    ROBOMIMIC_STEREO_LEROBOT,
    ROBOTWIN_STEREO_LEROBOT,
    STEREO_LEROBOT_DATASET_TYPES,
)
from .robomimic_stereo_lerobot import RobomimicStereoLeRobotDataset
from .robotwin_stereo_lerobot import (
    MultiRobotwinStereoLeRobotDataset,
    RobotwinStereoLeRobotDataset,
)


def require_stereo_lerobot_dataset_type(dataset_cfg: dict[str, Any]) -> str:
    dataset_type = dataset_cfg.get("type")
    if dataset_type not in STEREO_LEROBOT_DATASET_TYPES:
        valid = ", ".join(STEREO_LEROBOT_DATASET_TYPES)
        raise ValueError(f"dataset.type must be one of: {valid}. Got {dataset_type!r}.")
    return str(dataset_type)


def build_stereo_lerobot_dataset(
    dataset_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
) -> Dataset:
    dataset_type = require_stereo_lerobot_dataset_type(dataset_cfg)
    dataset_kwargs = {
        "camera_pairs": dataset_cfg["camera_pairs"],
        "num_history_frames": policy_cfg["num_history_frames"],
        "action_horizon": policy_cfg["action_horizon"],
        "image_size": dataset_cfg.get("image_size"),
        "episode_indices": dataset_cfg.get("episode_indices"),
        "num_episodes": dataset_cfg.get("num_episodes"),
        "action_normalization": dataset_cfg.get("action_normalization"),
    }

    if dataset_type == ROBOTWIN_STEREO_LEROBOT:
        if dataset_cfg.get("roots") is not None:
            return MultiRobotwinStereoLeRobotDataset(roots=dataset_cfg["roots"], **dataset_kwargs)
        return RobotwinStereoLeRobotDataset(root=dataset_cfg["root"], **dataset_kwargs)

    if dataset_cfg.get("roots") is not None:
        raise ValueError("robomimic-stereo-lerobot does not support dataset.roots.")
    return RobomimicStereoLeRobotDataset(root=dataset_cfg["root"], **dataset_kwargs)


__all__ = ["build_stereo_lerobot_dataset", "require_stereo_lerobot_dataset_type"]
