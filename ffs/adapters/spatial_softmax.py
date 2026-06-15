from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdapterOutput, BaseAdapter, feature_view_count, resolve_feature_names


class DynamicSpatialSoftmax(nn.Module):
    """Spatial softmax keypoint pooling for feature maps with dynamic H/W."""

    def __init__(
        self,
        in_channels: int,
        num_kp: int = 32,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_kp <= 0:
            raise ValueError("num_kp must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.num_kp = int(num_kp)
        self.noise_std = float(noise_std)
        self.heatmap = nn.Conv2d(int(in_channels), self.num_kp, kernel_size=1)
        self.kps: torch.Tensor | None = None
        temp = torch.tensor(float(temperature))
        if learnable_temperature:
            self.register_parameter("temperature", nn.Parameter(temp, requires_grad=True))
        else:
            self.register_buffer("temperature", temp)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feature.shape
        heatmap = self.heatmap(feature).flatten(2)
        temperature = self.temperature.to(device=feature.device, dtype=feature.dtype)
        attention = F.softmax(heatmap / temperature, dim=-1)

        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        pos_x = xx.flatten().view(1, 1, -1)
        pos_y = yy.flatten().view(1, 1, -1)
        expected_x = torch.sum(pos_x * attention, dim=-1)
        expected_y = torch.sum(pos_y * attention, dim=-1)
        keypoints = torch.stack([expected_x, expected_y], dim=-1)
        if self.training and self.noise_std > 0:
            keypoints = keypoints + torch.randn_like(keypoints) * self.noise_std
        self.kps = keypoints.detach()
        return keypoints.flatten(start_dim=1)


class SpatialSoftmaxAdapter(BaseAdapter):
    output_kind = "cond"

    def __init__(
        self,
        *,
        backbone: nn.Module,
        policy_cfg: dict[str, Any],
        dataset_cfg: dict[str, Any],
        feature_names: object = "auto",
        num_kp: int = 32,
        temperature: float = 1.0,
        noise_std: float = 0.0,
        learnable_temperature: bool = False,
        projection_dim: int | None = None,
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

        self.max_disp = float(getattr(backbone, "max_disp", 0) or 0)
        self.pools = nn.ModuleDict(
            {
                name: DynamicSpatialSoftmax(
                    in_channels=int(feature_channels[name]),
                    num_kp=int(num_kp),
                    temperature=float(temperature),
                    learnable_temperature=bool(learnable_temperature),
                    noise_std=float(noise_std),
                )
                for name in self.feature_names
            }
        )
        pooled_dim = int(num_kp) * 2
        self.projection_dim = int(projection_dim) if projection_dim is not None else None
        self.feature_dim = self.projection_dim or pooled_dim
        self.projections = nn.ModuleDict()
        if self.projection_dim is not None:
            self.projections = nn.ModuleDict(
                {name: nn.Linear(pooled_dim, self.projection_dim) for name in self.feature_names}
            )

        self.visual_dim_per_frame = sum(
            feature_view_count(backbone, name, self.num_stereo_pairs) * self.feature_dim
            for name in self.feature_names
        )
        self.cond_dim = self.num_history_frames * (self.visual_dim_per_frame + self.state_dim)

    def _pool_feature(
        self,
        name: str,
        feature: torch.Tensor,
        *,
        batch: int,
        time: int,
        views: int,
    ) -> torch.Tensor:
        if name == "disp":
            if self.max_disp <= 0:
                raise ValueError("Backbone must expose a positive max_disp when using 'disp' as a feature.")
            feature = (feature.float() / self.max_disp).clamp(0.0, 1.0).to(dtype=feature.dtype)

        if feature.ndim == 4:
            expected = batch * time * views
            if feature.shape[0] != expected:
                raise ValueError(f"Feature {name!r} expected batch {expected}, got {feature.shape[0]}.")
            flat = feature
            leading = (batch, time, views)
        elif feature.ndim == 5:
            if feature.shape[:2] != (batch, time):
                raise ValueError(f"Feature {name!r} expected leading [B,T], got {tuple(feature.shape[:2])}.")
            flat = feature.reshape(batch * time, *feature.shape[2:])
            leading = (batch, time, 1)
        elif feature.ndim == 6:
            if feature.shape[:3] != (batch, time, views):
                raise ValueError(f"Feature {name!r} expected leading [B,T,V], got {tuple(feature.shape[:3])}.")
            flat = feature.reshape(batch * time * views, *feature.shape[3:])
            leading = (batch, time, views)
        else:
            raise ValueError(f"SpatialSoftmaxAdapter expected map feature for {name!r}, got {tuple(feature.shape)}.")

        pooled = self.pools[name](flat)
        if name in self.projections:
            pooled = self.projections[name](pooled)
        return pooled.view(*leading, -1)

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
            raise ValueError("adapter.type='spatial_softmax' does not expose attention maps.")
        if time != self.num_history_frames:
            raise ValueError(f"Expected time={self.num_history_frames}, got {time}.")
        if state.shape[:2] != (batch, time) or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state shape [B,{time},{self.state_dim}], got {tuple(state.shape)}.")

        parts = []
        for name in self.feature_names:
            pooled = self._pool_feature(name, backbone_out[name], batch=batch, time=time, views=views)
            parts.append(pooled.flatten(start_dim=2))
        visual = torch.cat(parts, dim=-1)
        cond = torch.cat([visual, state.float()], dim=-1)
        return AdapterOutput(cond=cond.flatten(start_dim=1))


__all__ = ["DynamicSpatialSoftmax", "SpatialSoftmaxAdapter"]
