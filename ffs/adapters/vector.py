from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import AdapterOutput, BaseAdapter, feature_view_count, resolve_feature_names


class VectorAdapter(BaseAdapter):
    output_kind = "cond"

    def __init__(
        self,
        *,
        backbone: nn.Module,
        policy_cfg: dict[str, Any],
        dataset_cfg: dict[str, Any],
        feature_names: object = "auto",
    ) -> None:
        super().__init__()
        self.feature_names = resolve_feature_names(feature_names, backbone)
        self.num_history_frames = int(policy_cfg.get("observation_horizon", policy_cfg["num_history_frames"]))
        self.state_dim = int(policy_cfg["state_dim"])
        self.num_stereo_pairs = len(dataset_cfg.get("camera_pairs", []))
        if self.num_stereo_pairs <= 0:
            raise ValueError("dataset.camera_pairs must contain at least one stereo pair.")

        feature_channels = getattr(backbone, "feature_channels", None)
        if not isinstance(feature_channels, dict):
            raise ValueError("Backbone must expose a feature_channels dict.")
        missing = sorted(set(self.feature_names) - set(feature_channels))
        if missing:
            raise ValueError(f"Backbone is missing feature channels for: {', '.join(missing)}")

        self.feature_dims = {name: int(feature_channels[name]) for name in self.feature_names}
        self.visual_dim_per_frame = sum(
            feature_view_count(backbone, name, self.num_stereo_pairs) * self.feature_dims[name]
            for name in self.feature_names
        )
        self.cond_dim = self.num_history_frames * (self.visual_dim_per_frame + self.state_dim)

    def _flatten_feature(
        self,
        name: str,
        feature: torch.Tensor,
        *,
        batch: int,
        time: int,
        views: int,
    ) -> torch.Tensor:
        expected_dim = self.feature_dims[name]
        if feature.ndim == 4:
            expected = batch * time * views
            if feature.shape[0] != expected:
                raise ValueError(f"Feature {name!r} expected batch {expected}, got {feature.shape[0]}.")
            flat = feature.flatten(start_dim=1)
            if flat.shape[-1] != expected_dim:
                raise ValueError(f"Feature {name!r} expected dim {expected_dim}, got {flat.shape[-1]}.")
            return flat.view(batch, time, views, expected_dim)
        if feature.ndim == 2:
            expected = batch * time * views
            if feature.shape != (expected, expected_dim):
                raise ValueError(f"Feature {name!r} expected [{expected},{expected_dim}], got {tuple(feature.shape)}.")
            return feature.view(batch, time, views, expected_dim)
        if feature.ndim == 3:
            if feature.shape[:2] != (batch, time) or feature.shape[-1] != expected_dim:
                raise ValueError(f"Feature {name!r} expected [B,T,{expected_dim}], got {tuple(feature.shape)}.")
            return feature.unsqueeze(2)
        raise ValueError(f"VectorAdapter expected vector-like feature for {name!r}, got {tuple(feature.shape)}.")

    def forward(
        self,
        backbone_out: dict[str, torch.Tensor],
        state: torch.Tensor,
        *,
        batch: int,
        time: int,
        views: int,
        return_attention: bool = False,
    ) -> AdapterOutput:
        if return_attention:
            raise ValueError("adapter.type='vector' does not expose attention maps.")
        if time != self.num_history_frames:
            raise ValueError(f"Expected time={self.num_history_frames}, got {time}.")
        if state.shape[:2] != (batch, time) or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state shape [B,{time},{self.state_dim}], got {tuple(state.shape)}.")

        parts = [
            self._flatten_feature(name, backbone_out[name], batch=batch, time=time, views=views).flatten(start_dim=2)
            for name in self.feature_names
        ]
        visual = torch.cat(parts, dim=-1)
        cond = torch.cat([visual, state.float()], dim=-1)
        return AdapterOutput(cond=cond.flatten(start_dim=1))


__all__ = ["VectorAdapter"]
