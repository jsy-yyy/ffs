from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ffs.backbones.robomimic_cnn import RobomimicCNNBackbone
from ffs.heads.diffusion_unet import DiffusionUNetActionHead


class RobomimicDiffusionPolicy(nn.Module):
    def __init__(
        self,
        backbone: RobomimicCNNBackbone,
        action_head: DiffusionUNetActionHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.action_head = action_head
        self.state_dim = backbone.state_dim
        self.action_dim = action_head.action_dim
        self.observation_horizon = action_head.observation_horizon
        self.action_horizon = action_head.action_horizon
        self.prediction_horizon = action_head.prediction_horizon

    @staticmethod
    def split_state(state: torch.Tensor) -> dict[str, torch.Tensor]:
        return RobomimicCNNBackbone.split_state(state)

    def make_obs_dict(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone.make_obs_dict(left, right, state)

    def encode_obs(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.backbone(left, right, state)

    def training_loss(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        obs_cond = self.backbone(left, right, state)
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
        obs_cond = self.backbone(left, right, state)
        if action is not None:
            return self.action_head.training_loss(obs_cond, action)
        return self.action_head(obs_cond)


def build_robomimic_diffusion_policy(
    *,
    backbone_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    head_cfg: dict[str, Any],
    image_size: list[int] | tuple[int, int],
) -> RobomimicDiffusionPolicy:
    head_nested = dict(head_cfg.get("diffusion_unet") or {})
    ddpm_cfg = dict(head_nested.pop("ddpm", {}) or {})
    observation_horizon = int(policy_cfg.get("observation_horizon", policy_cfg["num_history_frames"]))
    backbone = RobomimicCNNBackbone(
        state_dim=int(policy_cfg["state_dim"]),
        observation_horizon=observation_horizon,
        image_size=backbone_cfg.get("image_size", image_size),
        use_left_only=bool(backbone_cfg.get("use_left_only", True)),
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
    return RobomimicDiffusionPolicy(backbone=backbone, action_head=action_head)


__all__ = ["RobomimicDiffusionPolicy", "build_robomimic_diffusion_policy"]
