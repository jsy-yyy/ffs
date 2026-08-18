from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ffs.adapters import AdapterOutput, BaseAdapter


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
        task_condition_enabled: bool = False,
        num_tasks: int = 0,
        task_to_id: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.action_head = action_head
        if (
            bool(getattr(adapter, "token_pruning_enabled", False))
            and getattr(adapter, "token_pruning_position_encoding", None) == "fixed_2d_sincos"
            and hasattr(action_head, "vision_pos_emb")
        ):
            action_head.vision_pos_emb.requires_grad_(False)
        self.num_history_frames = int(num_history_frames)
        self.state_dim = int(state_dim)
        self.num_stereo_pairs = int(num_stereo_pairs)
        self.disparity_provider = disparity_provider
        self.disparity_max_disp = float(disparity_max_disp or 0.0)
        self.disparity_ablation = str(disparity_ablation)
        self.task_condition_enabled = bool(task_condition_enabled)
        self.task_to_id = dict(task_to_id or {})
        self.id_to_task = tuple(
            task for task, _ in sorted(self.task_to_id.items(), key=lambda item: item[1])
        )
        self.num_tasks = int(num_tasks or len(self.task_to_id))
        if self.task_condition_enabled:
            if self.adapter.output_kind != "tokens":
                raise ValueError("task_condition requires a token adapter output.")
            if self.num_tasks <= 0:
                raise ValueError("task_condition requires num_tasks > 0.")
            self.task_embedding = nn.Embedding(self.num_tasks, int(adapter.token_dim))
        else:
            self.task_embedding = None
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

    def task_id_from_name(self, task_name: str | Sequence[str]) -> torch.Tensor | None:
        if not self.task_condition_enabled:
            return None
        if isinstance(task_name, str):
            names = [task_name]
        else:
            names = list(task_name)
        try:
            ids = [self.task_to_id[name] for name in names]
        except KeyError as exc:
            valid = ", ".join(self.id_to_task)
            raise KeyError(f"Unknown task name {exc.args[0]!r}; valid tasks: {valid}") from exc
        return torch.tensor(ids, dtype=torch.long)

    def _append_task_condition(
        self,
        tokens: torch.Tensor,
        adapter_out: AdapterOutput,
        task_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, AdapterOutput]:
        if not self.task_condition_enabled:
            return tokens, adapter_out
        if task_id is None:
            raise ValueError("task_id is required when policy.task_condition.enabled=true.")
        if self.task_embedding is None:
            raise RuntimeError("task_embedding is missing while task_condition is enabled.")
        task_id = task_id.to(device=tokens.device, dtype=torch.long).view(-1)
        if task_id.shape[0] != tokens.shape[0]:
            raise ValueError(f"task_id expected shape [{tokens.shape[0]}], got {tuple(task_id.shape)}.")
        if torch.any(task_id < 0) or torch.any(task_id >= self.num_tasks):
            raise ValueError(f"task_id values must be in [0, {self.num_tasks}).")

        task_token = self.task_embedding(task_id).to(dtype=tokens.dtype).unsqueeze(1)
        tokens = torch.cat([tokens, task_token], dim=1)

        condition_bias = adapter_out.condition_bias
        if condition_bias is not None:
            zeros = torch.zeros(tokens.shape[0], 1, device=condition_bias.device, dtype=condition_bias.dtype)
            condition_bias = torch.cat([condition_bias, zeros], dim=1)

        token_positions = adapter_out.token_positions
        if token_positions is not None:
            zeros = torch.zeros(tokens.shape[0], 1, 2, device=token_positions.device, dtype=token_positions.dtype)
            token_positions = torch.cat([token_positions, zeros], dim=1)

        return tokens, AdapterOutput(
            cond=adapter_out.cond,
            tokens=tokens,
            condition_bias=condition_bias,
            token_positions=token_positions,
            aux_loss=adapter_out.aux_loss,
            attention=adapter_out.attention,
        )

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

    def _encode_with_adapter_output(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        *,
        return_attention: bool = False,
        task_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AdapterOutput]:
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
            if self.task_condition_enabled:
                raise ValueError("task_condition is only supported with token adapter outputs.")
            if adapter_out.cond is None:
                raise ValueError("Adapter declared output_kind='cond' but returned no cond tensor.")
            return adapter_out.cond, adapter_out
        if adapter_out.tokens is None:
            raise ValueError("Adapter declared output_kind='tokens' but returned no tokens tensor.")
        return self._append_task_condition(adapter_out.tokens, adapter_out, task_id)

    def encode(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        *,
        return_attention: bool = False,
        task_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        encoded, adapter_out = self._encode_with_adapter_output(
            left,
            right,
            state,
            return_attention=return_attention,
            task_id=task_id,
        )
        return encoded, adapter_out.attention

    def training_loss(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded, adapter_out = self._encode_with_adapter_output(left, right, state, task_id=task_id)
        head_kwargs = {}
        if adapter_out.condition_bias is not None:
            head_kwargs["condition_bias"] = adapter_out.condition_bias
        if adapter_out.token_positions is not None:
            head_kwargs["token_positions"] = adapter_out.token_positions
        loss = self.action_head.training_loss(encoded, action, **head_kwargs)
        if adapter_out.aux_loss is not None:
            loss = loss + adapter_out.aux_loss.to(device=loss.device, dtype=loss.dtype)
        return loss

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        task_id: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if action is not None and return_attention:
            raise ValueError("return_attention is only supported for action prediction.")
        encoded, adapter_out = self._encode_with_adapter_output(
            left,
            right,
            state,
            return_attention=return_attention,
            task_id=task_id,
        )
        if action is not None:
            head_kwargs = {}
            if adapter_out.condition_bias is not None:
                head_kwargs["condition_bias"] = adapter_out.condition_bias
            if adapter_out.token_positions is not None:
                head_kwargs["token_positions"] = adapter_out.token_positions
            loss = self.action_head.training_loss(encoded, action, **head_kwargs)
            if adapter_out.aux_loss is not None:
                loss = loss + adapter_out.aux_loss.to(device=loss.device, dtype=loss.dtype)
            return loss
        head_kwargs = {}
        if adapter_out.condition_bias is not None:
            head_kwargs["condition_bias"] = adapter_out.condition_bias
        if adapter_out.token_positions is not None:
            head_kwargs["token_positions"] = adapter_out.token_positions
        action_pred = self.action_head(encoded, **head_kwargs)
        if return_attention:
            return action_pred, adapter_out.attention or {}
        return action_pred


__all__ = ["BackboneAdapterHeadPolicy"]
