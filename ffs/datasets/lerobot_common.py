from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


ROBOTWIN_STEREO_LEROBOT = "robotwin-stereo-lerobot"
ROBOMIMIC_STEREO_LEROBOT = "robomimic-stereo-lerobot"
STEREO_LEROBOT_DATASET_TYPES = (ROBOTWIN_STEREO_LEROBOT, ROBOMIMIC_STEREO_LEROBOT)


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


def _normalize_episode_indices(
    episode_indices: list[int] | tuple[int, ...] | set[int] | None,
) -> frozenset[int] | None:
    if episode_indices is None:
        return None
    normalized = frozenset(int(value) for value in episode_indices)
    if not normalized:
        raise ValueError("episode_indices must be non-empty when provided.")
    return normalized


def _normalize_num_episodes(num_episodes: int | None) -> int | None:
    if num_episodes is None:
        return None
    normalized = int(num_episodes)
    if normalized <= 0:
        raise ValueError("num_episodes must be positive when provided.")
    return normalized


def _validate_episode_filter(
    episode_indices: frozenset[int] | None,
    num_episodes: int | None,
) -> None:
    if episode_indices is not None and num_episodes is not None:
        raise ValueError("Specify only one of episode_indices or num_episodes.")


def _limit_rows_by_num_episodes(
    rows: list[dict[str, Any]],
    num_episodes: int | None,
) -> list[dict[str, Any]]:
    if num_episodes is None:
        return rows
    episode_ids = sorted({int(row["episode_index"]) for row in rows})
    keep = set(episode_ids[:num_episodes])
    return [row for row in rows if int(row["episode_index"]) in keep]


def _feature_shape(info: dict[str, Any], key: str) -> list[int]:
    try:
        shape = info["features"][key]["shape"]
    except KeyError as exc:
        raise ValueError(f"Dataset is missing required feature {key!r}") from exc
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"Expected {key!r} to be a 1D vector feature, got shape {shape!r}")
    return shape


def _resize_frame(frame: torch.Tensor, image_size: tuple[int, int] | None) -> torch.Tensor:
    if image_size is None:
        return frame
    return F.interpolate(
        frame.unsqueeze(0),
        size=image_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


__all__ = [
    "ActionNormalizer",
    "ROBOMIMIC_STEREO_LEROBOT",
    "ROBOTWIN_STEREO_LEROBOT",
    "STEREO_LEROBOT_DATASET_TYPES",
    "denormalize_action",
    "normalize_action",
]
