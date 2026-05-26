from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ffs.backbones import FoundationStereoBackbone
from ffs.heads import ActionHead, build_action_head


class StereoActionPolicy(nn.Module):
    """FoundationStereo visual history + proprioceptive history -> actions."""

    def __init__(
        self,
        backbone: nn.Module,
        num_history_frames: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        num_stereo_pairs: int,
        use_disparity: bool = True,
        condition_token_dim: int = 256,
        disp_grid_size: Sequence[int] = (4, 5),
        head_hidden_dim: int | None = None,
        head_layers: int | None = None,
        head_dropout: float | None = None,
        action_head_cfg: dict[str, Any] | None = None,
        action_head: ActionHead | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_history_frames = num_history_frames
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_stereo_pairs = num_stereo_pairs
        self.use_disparity = use_disparity
        self.condition_token_dim = condition_token_dim
        self.disp_grid_size = self._validate_disp_grid_size(disp_grid_size)
        self.n_disp_tokens = self.disp_grid_size[0] * self.disp_grid_size[1]
        self.feature_names = tuple(backbone.feature_names)

        if condition_token_dim <= 0:
            raise ValueError("condition_token_dim must be positive.")

        feature_channels = getattr(backbone, "feature_channels", None)
        if not isinstance(feature_channels, dict):
            raise ValueError("Backbone must expose a feature_channels dict.")
        missing_channels = sorted(set(self.feature_names) - set(feature_channels))
        if missing_channels:
            raise ValueError(f"Backbone is missing feature channels for: {', '.join(missing_channels)}")

        self.visual_projections = nn.ModuleDict(
            {
                name: nn.Linear(int(feature_channels[name]), condition_token_dim)
                for name in self.feature_names
            }
        )
        self.state_proj = nn.Linear(state_dim, condition_token_dim)

        if use_disparity:
            self.disp_encoder = nn.Sequential(
                nn.Conv2d(1, condition_token_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(self.disp_grid_size),
                nn.Flatten(start_dim=2),
            )
        else:
            self.disp_encoder = None

        tokens_per_frame = len(self.feature_names) * num_stereo_pairs + 1
        if use_disparity:
            tokens_per_frame += self.n_disp_tokens * num_stereo_pairs
        condition_len = num_history_frames * tokens_per_frame

        if action_head is not None and action_head_cfg is not None:
            raise ValueError("Pass either action_head or action_head_cfg, not both.")
        legacy_head_cfg = {
            key: value
            for key, value in {
                "hidden_dim": head_hidden_dim,
                "num_blocks": head_layers,
                "dropout": head_dropout,
            }.items()
            if value is not None
        }
        if action_head_cfg is not None and legacy_head_cfg:
            raise ValueError("Pass either action_head_cfg or legacy head_* args, not both.")
        if action_head_cfg is None and legacy_head_cfg:
            action_head_cfg = legacy_head_cfg

        self.action_head = action_head or build_action_head(
            action_head_cfg,
            input_dim=condition_len * condition_token_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            frame_token_dim=condition_token_dim,
            condition_len=condition_len,
        )

    def _validate_disp_grid_size(self, disp_grid_size: Sequence[int]) -> tuple[int, int]:
        grid = tuple(int(value) for value in disp_grid_size)
        if len(grid) != 2:
            raise ValueError("disp_grid_size must contain exactly two integers.")
        if grid[0] <= 0 or grid[1] <= 0:
            raise ValueError("disp_grid_size values must be positive.")
        return grid

    def _pool_visual_tokens(self, backbone_out: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for name in self.feature_names:
            feat = backbone_out[name]
            token = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            tokens.append(self.visual_projections[name](token))
        return torch.stack(tokens, dim=1)

    def encode_tokens(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b, t, v, c, h, w = left.shape
        left = left.reshape(b * t * v, c, h, w)
        right = right.reshape(b * t * v, c, h, w)

        backbone_out = self.backbone(left, right)
        visual_tokens = self._pool_visual_tokens(backbone_out)
        visual_tokens = visual_tokens.view(b, t, v, len(self.feature_names), -1).flatten(2, 3)
        frame_parts = [visual_tokens]

        if self.use_disparity:
            # disp: [B*T*V, 1, H, W] -> row-major grid tokens: [B, T, V*n_disp_tokens, C]
            disp_tokens = self.disp_encoder(backbone_out["disp"].float()).transpose(1, 2)
            disp_tokens = disp_tokens.view(b, t, v, self.n_disp_tokens, -1).flatten(2, 3)
            frame_parts.append(disp_tokens)

        state_tokens = self.state_proj(state.reshape(b, t, self.state_dim)).unsqueeze(2)
        frame_parts.append(state_tokens)
        return torch.cat(frame_parts, dim=2).flatten(1, 2)

    def training_loss(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_head.training_loss(self.encode_tokens(left, right, state), action)

    def forward(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.encode_tokens(left, right, state))


def resolve_action_head_cfg(head_cfg: dict[str, Any]) -> dict[str, Any]:
    head_type = head_cfg.get("type", "mlp")
    nested_cfg = head_cfg.get(head_type)
    if isinstance(nested_cfg, dict):
        return {"type": head_type, **nested_cfg}

    reserved_keys = {"condition_token_dim", "disp_grid_size", "mlp", "rdt"}
    return {k: v for k, v in head_cfg.items() if k not in reserved_keys}


def build_policy(cfg: dict[str, Any]) -> StereoActionPolicy:
    backbone_cfg = cfg["backbone"]
    policy_cfg = cfg["policy"]
    head_cfg = cfg.get("head", {})
    dataset_cfg = cfg.get("dataset", {})
    action_head_cfg = resolve_action_head_cfg(head_cfg)
    num_stereo_pairs = len(dataset_cfg.get("camera_pairs", []))

    backbone = FoundationStereoBackbone(
        foundation_root=backbone_cfg["foundation_root"],
        checkpoint_path=backbone_cfg["checkpoint_path"],
        valid_iters=backbone_cfg.get("valid_iters", 4),
        max_disp=backbone_cfg.get("max_disp", 192),
        freeze=backbone_cfg.get("freeze", True),
        use_disparity=policy_cfg.get("use_disparity", True),
        optimize_build_volume=backbone_cfg.get("optimize_build_volume", "pytorch1"),
        feature_names=backbone_cfg["feature_names"],
    )

    return StereoActionPolicy(
        backbone=backbone,
        num_history_frames=policy_cfg["num_history_frames"],
        state_dim=policy_cfg["state_dim"],
        action_dim=policy_cfg["action_dim"],
        action_horizon=policy_cfg["action_horizon"],
        num_stereo_pairs=num_stereo_pairs,
        use_disparity=policy_cfg.get("use_disparity", True),
        condition_token_dim=head_cfg["condition_token_dim"],
        disp_grid_size=head_cfg["disp_grid_size"],
        action_head_cfg=action_head_cfg,
    )
