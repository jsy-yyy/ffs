from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ffs.backbones.robomimic_cnn import RobomimicCNNBackbone
from ffs.heads.diffusion_unet import DiffusionUNetActionHead


class RobomimicDiffusionPolicy(nn.Module):
    def __init__(
        self,
        backbone: RobomimicCNNBackbone,
        action_head: DiffusionUNetActionHead,
        disparity_provider: nn.Module | None = None,
        disparity_max_disp: int | float | None = None,
        disparity_ablation: str = "none",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.action_head = action_head
        self.disparity_provider = disparity_provider
        self.disparity_max_disp = float(disparity_max_disp or 0.0)
        self.disparity_ablation = str(disparity_ablation)
        if self.disparity_provider is not None and self.disparity_max_disp <= 0:
            self.disparity_max_disp = float(getattr(self.disparity_provider, "max_disp", 0) or 0)
        if self.disparity_provider is not None and self.disparity_max_disp <= 0:
            raise ValueError("disparity_max_disp must be positive when a disparity_provider is configured.")
        if self.disparity_ablation not in {"none", "zero", "shuffle"}:
            raise ValueError("disparity_ablation must be one of: none, zero, shuffle.")
        self.state_dim = backbone.state_dim
        self.action_dim = action_head.action_dim
        self.observation_horizon = action_head.observation_horizon
        self.action_horizon = action_head.action_horizon
        self.prediction_horizon = action_head.prediction_horizon

    @staticmethod
    def split_state(state: torch.Tensor) -> dict[str, torch.Tensor]:
        return RobomimicCNNBackbone.split_state(state)

    def make_obs_dict(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        disparity = self._compute_disparity(left, right)
        return self.backbone.make_obs_dict(left, right, state, disparity)

    def encode_obs(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        disparity = self._compute_disparity(left, right)
        return self.backbone(left, right, state, disparity)

    def _ablate_disparity(self, disparity: torch.Tensor) -> torch.Tensor:
        if self.disparity_ablation == "none":
            return disparity
        if self.disparity_ablation == "zero":
            return torch.zeros_like(disparity)
        if self.disparity_ablation == "shuffle":
            spatial_size = disparity.shape[-2] * disparity.shape[-1]
            if spatial_size <= 1:
                return disparity
            stride = spatial_size - 1
            offset = spatial_size // 3 + 1
            perm = (torch.arange(spatial_size, device=disparity.device) * stride + offset) % spatial_size
            return disparity.flatten(2).index_select(2, perm).view_as(disparity)
        raise ValueError(f"Unsupported disparity_ablation: {self.disparity_ablation}")

    def _compute_disparity(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor | None:
        if self.disparity_provider is None:
            return None
        if left.ndim != 6 or right.ndim != 6:
            raise ValueError(
                "RobomimicDiffusionPolicy expects left/right shape [B,T,V,3,H,W], "
                f"got left={tuple(left.shape)} right={tuple(right.shape)}."
            )
        if left.shape != right.shape:
            raise ValueError(f"left/right image shapes must match, got {tuple(left.shape)} and {tuple(right.shape)}.")
        batch, time, views, channels, height, width = left.shape
        left_flat = left.reshape(batch * time * views, channels, height, width)
        right_flat = right.reshape(batch * time * views, channels, height, width)
        provider_out = self.disparity_provider(left_flat, right_flat)
        if not isinstance(provider_out, dict) or "disp" not in provider_out:
            raise ValueError("disparity_provider must return a dict containing 'disp'.")
        disparity = provider_out["disp"]
        if disparity.ndim == 3:
            disparity = disparity.unsqueeze(1)
        if disparity.ndim != 4 or disparity.shape[1] != 1:
            raise ValueError(f"Expected provider disparity shape [N,1,H,W], got {tuple(disparity.shape)}.")
        disparity = (disparity.float() / self.disparity_max_disp).clamp(0.0, 1.0)
        disparity = self._ablate_disparity(disparity)
        if disparity.shape[-2:] != (height, width):
            disparity = F.interpolate(disparity, size=(height, width), mode="bilinear", align_corners=False)
        return disparity.view(batch, time, views, 1, height, width)

    def training_loss(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        obs_cond = self.encode_obs(left, right, state)
        return self.action_head.training_loss(obs_cond, action)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        if return_attention:
            raise ValueError("robomimic_diffusion does not expose query attention maps.")
        obs_cond = self.encode_obs(left, right, state)
        if action is not None:
            return self.action_head.training_loss(obs_cond, action)
        return self.action_head(obs_cond)


class DynamicSpatialSoftmax(nn.Module):
    """Spatial softmax keypoint pooling for feature maps with dynamic H/W."""

    def __init__(self, in_channels: int, num_kp: int = 32, temperature: float = 1.0) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_kp <= 0:
            raise ValueError("num_kp must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.num_kp = int(num_kp)
        self.heatmap = nn.Conv2d(int(in_channels), self.num_kp, kernel_size=1)
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feature.shape
        heatmap = self.heatmap(feature).flatten(2)
        attention = F.softmax(heatmap / self.temperature.to(device=feature.device, dtype=feature.dtype), dim=-1)

        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        pos_x = xx.flatten().view(1, 1, -1)
        pos_y = yy.flatten().view(1, 1, -1)
        expected_x = torch.sum(pos_x * attention, dim=-1)
        expected_y = torch.sum(pos_y * attention, dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1).flatten(start_dim=1)


class SpatialBackboneDiffusionPolicy(nn.Module):
    """Spatial stereo backbone + SpatialSoftmax features -> diffusion action head."""

    def __init__(
        self,
        backbone: nn.Module,
        action_head: DiffusionUNetActionHead,
        *,
        num_history_frames: int,
        state_dim: int,
        num_stereo_pairs: int,
        num_kp: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_history_frames <= 0:
            raise ValueError("num_history_frames must be positive.")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if num_stereo_pairs <= 0:
            raise ValueError("num_stereo_pairs must be positive.")

        self.backbone = backbone
        self.action_head = action_head
        self.num_history_frames = int(num_history_frames)
        self.state_dim = int(state_dim)
        self.num_stereo_pairs = int(num_stereo_pairs)
        self.feature_names = tuple(backbone.feature_names)
        self.action_dim = action_head.action_dim
        self.observation_horizon = action_head.observation_horizon
        self.action_horizon = action_head.action_horizon
        self.prediction_horizon = action_head.prediction_horizon

        feature_channels = getattr(backbone, "feature_channels", None)
        if not isinstance(feature_channels, dict):
            raise ValueError("Backbone must expose a feature_channels dict.")
        missing_channels = sorted(set(self.feature_names) - set(feature_channels))
        if missing_channels:
            raise ValueError(f"Backbone is missing feature channels for: {', '.join(missing_channels)}")

        self.spatial_pools = nn.ModuleDict(
            {
                name: DynamicSpatialSoftmax(
                    in_channels=int(feature_channels[name]),
                    num_kp=num_kp,
                    temperature=temperature,
                )
                for name in self.feature_names
            }
        )
        self.visual_dim_per_pair = len(self.feature_names) * int(num_kp) * 2
        self.output_dim = (
            self.num_history_frames
            * (self.num_stereo_pairs * self.visual_dim_per_pair + self.state_dim)
        )
        if self.output_dim != action_head.cond_dim:
            raise ValueError(
                f"SpatialBackboneDiffusionPolicy cond dim mismatch: encoder={self.output_dim}, "
                f"action_head={action_head.cond_dim}."
            )

    def encode_obs(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch, time, views, channels, height, width = left.shape
        if time != self.num_history_frames:
            raise ValueError(
                f"SpatialBackboneDiffusionPolicy expected num_history_frames={self.num_history_frames}, got {time}."
            )
        if views != self.num_stereo_pairs:
            raise ValueError(
                f"SpatialBackboneDiffusionPolicy expected num_stereo_pairs={self.num_stereo_pairs}, got {views}."
            )
        if state.shape[:2] != (batch, time) or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state shape [B,{time},{self.state_dim}], got {tuple(state.shape)}."
            )

        left_flat = left.reshape(batch * time * views, channels, height, width)
        right_flat = right.reshape(batch * time * views, channels, height, width)
        backbone_out = self.backbone(left_flat, right_flat)

        feature_parts = []
        for name in self.feature_names:
            feature = backbone_out[name]
            if name == "disp":
                max_disp = float(getattr(self.backbone, "max_disp", 0) or 0)
                if max_disp <= 0:
                    raise ValueError("Backbone must expose a positive max_disp when using 'disp' as a feature.")
                feature = (feature.float() / max_disp).clamp(0.0, 1.0).to(dtype=feature.dtype)
            pooled = self.spatial_pools[name](feature)
            pooled = pooled.view(batch, time, views, -1)
            feature_parts.append(pooled)
        visual = torch.cat(feature_parts, dim=-1).flatten(start_dim=2)
        cond = torch.cat([visual, state.float()], dim=-1)
        return cond.flatten(start_dim=1)

    def training_loss(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        obs_cond = self.encode_obs(left, right, state)
        return self.action_head.training_loss(obs_cond, action)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        if return_attention:
            raise ValueError("spatial_backbone_diffusion does not expose query attention maps.")
        obs_cond = self.encode_obs(left, right, state)
        if action is not None:
            return self.action_head.training_loss(obs_cond, action)
        return self.action_head(obs_cond)


def build_robomimic_diffusion_policy(
    *,
    backbone_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    head_cfg: dict[str, Any],
    image_size: list[int] | tuple[int, int],
    disparity_provider: nn.Module | None = None,
    disparity_max_disp: int | float | None = None,
) -> RobomimicDiffusionPolicy:
    head_nested = dict(head_cfg.get("diffusion_unet") or {})
    ddpm_cfg = dict(head_nested.pop("ddpm", {}) or {})
    observation_horizon = int(policy_cfg.get("observation_horizon", policy_cfg["num_history_frames"]))
    backbone = RobomimicCNNBackbone(
        state_dim=int(policy_cfg["state_dim"]),
        observation_horizon=observation_horizon,
        image_size=backbone_cfg.get("image_size", image_size),
        use_left_only=bool(backbone_cfg.get("use_left_only", True)),
        use_disparity=disparity_provider is not None,
        rgb_cfg=backbone_cfg.get("rgb"),
    )
    action_head = DiffusionUNetActionHead(
        cond_dim=backbone.output_dim,
        action_dim=int(policy_cfg["action_dim"]),
        observation_horizon=observation_horizon,
        action_horizon=int(policy_cfg["action_horizon"]),
        prediction_horizon=int(policy_cfg.get("prediction_horizon", policy_cfg["action_horizon"])),
        diffusion_step_embed_dim=int(head_nested.get("diffusion_step_embed_dim", 256)),
        down_dims=head_nested.get("down_dims", [256, 512, 1024]),
        kernel_size=int(head_nested.get("kernel_size", 5)),
        n_groups=int(head_nested.get("n_groups", 8)),
        ddpm_cfg=ddpm_cfg,
    )
    return RobomimicDiffusionPolicy(
        backbone=backbone,
        action_head=action_head,
        disparity_provider=disparity_provider,
        disparity_max_disp=disparity_max_disp,
        disparity_ablation=policy_cfg.get("disparity_ablation", "none"),
    )


def build_spatial_backbone_diffusion_policy(
    *,
    backbone: nn.Module,
    policy_cfg: dict[str, Any],
    head_cfg: dict[str, Any],
    num_stereo_pairs: int,
) -> SpatialBackboneDiffusionPolicy:
    head_nested = dict(head_cfg.get("diffusion_unet") or {})
    ddpm_cfg = dict(head_nested.pop("ddpm", {}) or {})
    pool_cfg = dict(head_nested.pop("spatial_softmax", {}) or {})
    observation_horizon = int(policy_cfg.get("observation_horizon", policy_cfg["num_history_frames"]))
    num_kp = int(pool_cfg.get("num_kp", 32))
    cond_dim = observation_horizon * (
        int(num_stereo_pairs) * len(tuple(backbone.feature_names)) * num_kp * 2
        + int(policy_cfg["state_dim"])
    )
    action_head = DiffusionUNetActionHead(
        cond_dim=cond_dim,
        action_dim=int(policy_cfg["action_dim"]),
        observation_horizon=observation_horizon,
        action_horizon=int(policy_cfg["action_horizon"]),
        prediction_horizon=int(policy_cfg.get("prediction_horizon", policy_cfg["action_horizon"])),
        diffusion_step_embed_dim=int(head_nested.get("diffusion_step_embed_dim", 256)),
        down_dims=head_nested.get("down_dims", [256, 512, 1024]),
        kernel_size=int(head_nested.get("kernel_size", 5)),
        n_groups=int(head_nested.get("n_groups", 8)),
        ddpm_cfg=ddpm_cfg,
    )
    return SpatialBackboneDiffusionPolicy(
        backbone=backbone,
        action_head=action_head,
        num_history_frames=observation_horizon,
        state_dim=int(policy_cfg["state_dim"]),
        num_stereo_pairs=int(num_stereo_pairs),
        num_kp=num_kp,
        temperature=float(pool_cfg.get("temperature", 1.0)),
    )


__all__ = [
    "DynamicSpatialSoftmax",
    "RobomimicDiffusionPolicy",
    "SpatialBackboneDiffusionPolicy",
    "build_robomimic_diffusion_policy",
    "build_spatial_backbone_diffusion_policy",
]
