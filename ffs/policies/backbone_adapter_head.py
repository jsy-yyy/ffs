from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ffs.adapters import BaseAdapter


class BackboneAdapterHeadPolicy(nn.Module):
    """Generic backbone -> adapter -> action head policy."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        adapter: BaseAdapter,
        action_head: nn.Module,
        num_history_frames: int,
        state_dim: int,
        num_stereo_pairs: int,
        disparity_provider: nn.Module | None = None,
        disparity_max_disp: int | float | None = None,
        disparity_ablation: str = "none",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.action_head = action_head
        self.num_history_frames = int(num_history_frames)
        self.state_dim = int(state_dim)
        self.num_stereo_pairs = int(num_stereo_pairs)
        self.disparity_provider = disparity_provider
        self.disparity_max_disp = float(disparity_max_disp or 0.0)
        self.disparity_ablation = str(disparity_ablation)
        if self.disparity_provider is not None and self.disparity_max_disp <= 0:
            self.disparity_max_disp = float(getattr(self.disparity_provider, "max_disp", 0) or 0)
        if self.disparity_provider is not None and self.disparity_max_disp <= 0:
            raise ValueError("disparity_max_disp must be positive when a disparity_provider is configured.")
        if self.disparity_ablation not in {"none", "zero", "shuffle"}:
            raise ValueError("disparity_ablation must be one of: none, zero, shuffle.")

        self.feature_names = tuple(getattr(backbone, "feature_names", ()))
        self.action_dim = action_head.action_dim
        self.observation_horizon = getattr(action_head, "observation_horizon", self.num_history_frames)
        self.action_horizon = action_head.action_horizon
        self.prediction_horizon = getattr(action_head, "prediction_horizon", self.action_horizon)

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

    def _run_backbone(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, time, views, channels, height, width = left.shape
        if getattr(self.backbone, "expects_sequence_input", False):
            disparity = self._compute_disparity(left, right)
            return self.backbone(left, right, state, disparity)

        left_flat = left.reshape(batch * time * views, channels, height, width)
        right_flat = right.reshape(batch * time * views, channels, height, width)
        if getattr(self.backbone, "uses_view_indices", False):
            view_indices = torch.arange(views, device=left.device).view(1, 1, views).expand(batch, time, views)
            return self.backbone(left_flat, right_flat, view_indices=view_indices.reshape(batch * time * views))
        return self.backbone(left_flat, right_flat)

    def encode(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if left.ndim != 6 or right.ndim != 6:
            raise ValueError(
                "BackboneAdapterHeadPolicy expects left/right shape [B,T,V,3,H,W], "
                f"got left={tuple(left.shape)} right={tuple(right.shape)}."
            )
        if left.shape != right.shape:
            raise ValueError(f"left/right image shapes must match, got {tuple(left.shape)} and {tuple(right.shape)}.")
        batch, time, views = left.shape[:3]
        if time != self.num_history_frames:
            raise ValueError(f"Expected num_history_frames={self.num_history_frames}, got {time}.")
        if views != self.num_stereo_pairs:
            raise ValueError(f"Expected num_stereo_pairs={self.num_stereo_pairs}, got {views}.")
        if state.shape[:2] != (batch, time) or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state shape [B,{time},{self.state_dim}], got {tuple(state.shape)}.")

        backbone_out = self._run_backbone(left, right, state)
        adapter_out = self.adapter(
            backbone_out,
            state,
            batch=batch,
            time=time,
            views=views,
            return_attention=return_attention,
        )
        if self.adapter.output_kind == "cond":
            if adapter_out.cond is None:
                raise ValueError("Adapter declared output_kind='cond' but returned no cond tensor.")
            return adapter_out.cond, adapter_out.attention
        if adapter_out.tokens is None:
            raise ValueError("Adapter declared output_kind='tokens' but returned no tokens tensor.")
        return adapter_out.tokens, adapter_out.attention

    def training_loss(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        encoded, _ = self.encode(left, right, state)
        return self.action_head.training_loss(encoded, action)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if action is not None and return_attention:
            raise ValueError("return_attention is only supported for action prediction.")
        encoded, attention = self.encode(left, right, state, return_attention=return_attention)
        if action is not None:
            return self.action_head.training_loss(encoded, action)
        action_pred = self.action_head(encoded)
        if return_attention:
            return action_pred, attention or {}
        return action_pred


__all__ = ["BackboneAdapterHeadPolicy"]
