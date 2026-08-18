from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdapterOutput, BaseAdapter, feature_view_count, resolve_feature_names


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _image_size(dataset_cfg: dict[str, Any]) -> tuple[int, int]:
    size = dataset_cfg.get("image_size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("dataset.image_size must be set to [H, W] for adapter.type='reshape_tokens'.")
    return int(size[0]), int(size[1])


def _robomimic_feature_hw(backbone: nn.Module, name: str) -> tuple[int, int] | None:
    obs_encoder = getattr(backbone, "obs_encoder", None)
    randomizers = getattr(obs_encoder, "obs_randomizers", None)
    if randomizers is None or name not in randomizers:
        return None
    modules = randomizers[name]
    if len(modules) <= 0:
        return None
    randomizer = modules[0]
    crop_height = getattr(randomizer, "crop_height", None)
    crop_width = getattr(randomizer, "crop_width", None)
    if crop_height is None or crop_width is None:
        return None
    return _ceil_div(int(crop_height), 32), _ceil_div(int(crop_width), 32)


def _feature_hw(backbone: nn.Module, name: str, dataset_cfg: dict[str, Any]) -> tuple[int, int]:
    explicit_shapes = getattr(backbone, "feature_spatial_shapes", None)
    if isinstance(explicit_shapes, dict) and name in explicit_shapes:
        shape = explicit_shapes[name]
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(f"Backbone feature_spatial_shapes[{name!r}] must be [H, W].")
        return int(shape[0]), int(shape[1])

    robomimic_hw = _robomimic_feature_hw(backbone, name)
    if robomimic_hw is not None:
        return robomimic_hw

    height, width = _image_size(dataset_cfg)
    if name == "stereo_latent":
        return 1, 1
    if name == "disp":
        return height, width
    if name in {"feat_04", "feat_08", "feat_16", "feat_32"}:
        stride = int(name.rsplit("_", maxsplit=1)[-1])
        return max(1, height // stride), max(1, width // stride)
    if name == "dino":
        dino = getattr(backbone, "dino", None)
        patch_size = int(getattr(dino, "patch_size", 14) or 14)
        return max(height // patch_size, 1), max(width // patch_size, 1)
    if name in {"fmap1", "net", "refine_net"}:
        return _ceil_div(height, 2), _ceil_div(width, 2)
    if name == "delta_block12":
        height_2 = _ceil_div(height, 2)
        width_2 = _ceil_div(width, 2)
        out_h = height_2 // 8
        out_w = width_2 // 8
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "WAFT delta_block12 needs image_size large enough for a positive "
                f"ceil(H/2)//8 x ceil(W/2)//8 grid, got image_size={[height, width]}."
            )
        return out_h, out_w

    raise ValueError(
        f"Cannot infer spatial shape for feature {name!r} in adapter.type='reshape_tokens'. "
        "Use a known backbone feature name or expose backbone.feature_spatial_shapes."
    )


class CoarseSaliencyHead(nn.Module):
    """Predict coarse saliency logits from a dense feature map."""

    def __init__(self, in_channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_channels or in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(feature, output_size)
        return self.net(pooled)


class ReshapeTokensAdapter(BaseAdapter):
    """Parameter-free adapter that reshapes backbone features into condition tokens."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        policy_cfg: dict[str, Any],
        dataset_cfg: dict[str, Any],
        feature_names: object = "auto",
        flatten: bool = False,
        saliency: dict[str, Any] | None = None,
        token_pruning: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.feature_names = resolve_feature_names(feature_names, backbone)
        if not self.feature_names:
            raise ValueError("adapter.feature_names must select at least one feature.")

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

        channels = {name: int(feature_channels[name]) for name in self.feature_names}
        unique_channels = sorted(set(channels.values()))
        if len(unique_channels) != 1:
            detail = ", ".join(f"{name}={channels[name]}" for name in self.feature_names)
            raise ValueError(
                "adapter.type='reshape_tokens' requires all selected features to have the same channel count; "
                f"got {detail}."
            )

        self.token_dim = unique_channels[0]
        if self.state_dim > self.token_dim:
            raise ValueError(
                "adapter.type='reshape_tokens' requires state_dim <= token_dim for zero-padded state tokens; "
                f"got state_dim={self.state_dim}, token_dim={self.token_dim}."
            )

        self.feature_hw = {
            name: _feature_hw(backbone, name, dataset_cfg)
            for name in self.feature_names
        }
        self.feature_token_counts = {
            name: height * width
            for name, (height, width) in self.feature_hw.items()
        }
        self.feature_view_counts = {
            name: feature_view_count(backbone, name, self.num_stereo_pairs)
            for name in self.feature_names
        }
        self.flatten = bool(flatten)
        self.output_kind = "cond" if self.flatten else "tokens"

        saliency_cfg = dict(saliency or {})
        self.saliency_enabled = bool(saliency_cfg.get("enabled", False))
        self.saliency_source_feature = str(saliency_cfg.get("source_feature", "refine_net"))
        self.saliency_mode = str(saliency_cfg.get("mode", "attention_bias"))
        self.saliency_bias_strength = float(saliency_cfg.get("bias_strength", 2.0))
        self.saliency_keep_ratio = float(saliency_cfg.get("keep_ratio", 0.10))
        self.saliency_sparsity_loss_weight = float(saliency_cfg.get("sparsity_loss_weight", 1.0e-3))
        self.saliency_entropy_loss_weight = float(saliency_cfg.get("entropy_loss_weight", 1.0e-4))
        self.saliency_head: CoarseSaliencyHead | None = None
        if self.saliency_enabled:
            if self.flatten:
                raise ValueError("adapter.saliency requires flatten=false so condition_bias can be passed to RDT.")
            if self.saliency_mode != "attention_bias":
                raise ValueError("adapter.saliency.mode currently supports only 'attention_bias'.")
            if self.saliency_source_feature not in self.feature_names:
                raise ValueError("adapter.saliency.source_feature must be included in adapter.feature_names.")
            if not 0.0 < self.saliency_keep_ratio < 1.0:
                raise ValueError("adapter.saliency.keep_ratio must be in (0, 1).")
            grid_size = saliency_cfg.get("grid_size", [14, 14])
            if not isinstance(grid_size, (list, tuple)) or len(grid_size) != 2:
                raise ValueError("adapter.saliency.grid_size must be [H, W].")
            self.saliency_grid_size = (int(grid_size[0]), int(grid_size[1]))
            if self.saliency_grid_size[0] <= 0 or self.saliency_grid_size[1] <= 0:
                raise ValueError("adapter.saliency.grid_size values must be positive.")
            hidden_channels = saliency_cfg.get("hidden_channels")
            self.saliency_head = CoarseSaliencyHead(
                in_channels=channels[self.saliency_source_feature],
                hidden_channels=int(hidden_channels) if hidden_channels is not None else None,
            )
        else:
            self.saliency_grid_size = (1, 1)

        self.full_visual_tokens_per_frame = sum(
            self.feature_view_counts[name] * self.feature_token_counts[name]
            for name in self.feature_names
        )
        pruning_cfg = dict(token_pruning or {})
        self.token_pruning_enabled = bool(pruning_cfg.get("enabled", False))
        self.token_pruning_source = str(pruning_cfg.get("source", "saliency"))
        self.token_pruning_keep_ratio = float(pruning_cfg.get("keep_ratio", self.saliency_keep_ratio))
        self.token_pruning_position_encoding = str(pruning_cfg.get("position_encoding", "fixed_2d_sincos"))
        self.pruned_visual_tokens_per_frame = self.full_visual_tokens_per_frame
        if self.token_pruning_enabled:
            if self.flatten:
                raise ValueError("adapter.token_pruning requires flatten=false.")
            if not self.saliency_enabled:
                raise ValueError("adapter.token_pruning source='saliency' requires adapter.saliency.enabled=true.")
            if self.token_pruning_source != "saliency":
                raise ValueError("adapter.token_pruning.source currently supports only 'saliency'.")
            if self.token_pruning_position_encoding != "fixed_2d_sincos":
                raise ValueError(
                    "adapter.token_pruning.position_encoding currently supports only 'fixed_2d_sincos'."
                )
            if len(self.feature_names) != 1 or self.feature_names[0] != self.saliency_source_feature:
                raise ValueError(
                    "adapter.token_pruning currently supports a single source feature matching "
                    "adapter.saliency.source_feature."
                )
            if not 0.0 < self.token_pruning_keep_ratio <= 1.0:
                raise ValueError("adapter.token_pruning.keep_ratio must be in (0, 1].")
            self.pruned_visual_tokens_per_frame = max(
                1,
                min(
                    self.full_visual_tokens_per_frame,
                    int(math.ceil(self.full_visual_tokens_per_frame * self.token_pruning_keep_ratio)),
                ),
            )

        self.tokens_per_frame = self.pruned_visual_tokens_per_frame + 1
        self.condition_len = self.num_history_frames * self.tokens_per_frame
        self.cond_dim = self.condition_len * self.token_dim

    def _check_channels(self, name: str, channels: int) -> None:
        if int(channels) != self.token_dim:
            raise ValueError(
                f"Feature {name!r} expected channel dim {self.token_dim}, got {int(channels)}."
            )

    def _feature_to_tokens(
        self,
        name: str,
        feature: torch.Tensor,
        *,
        batch: int,
        time: int,
        views: int,
    ) -> torch.Tensor:
        if feature.ndim == 6:
            if feature.shape[:3] != (batch, time, views):
                raise ValueError(f"Feature {name!r} expected leading [B,T,V], got {tuple(feature.shape[:3])}.")
            self._check_channels(name, feature.shape[3])
            return feature.permute(0, 1, 2, 4, 5, 3).reshape(batch, time, -1, self.token_dim)

        if feature.ndim == 5:
            if feature.shape[:2] != (batch, time):
                raise ValueError(f"Feature {name!r} expected leading [B,T], got {tuple(feature.shape[:2])}.")
            self._check_channels(name, feature.shape[2])
            return feature.permute(0, 1, 3, 4, 2).reshape(batch, time, -1, self.token_dim)

        if feature.ndim == 4:
            self._check_channels(name, feature.shape[1])
            if feature.shape[0] == batch * time * views:
                leading = (batch, time, views)
            elif feature.shape[0] == batch * time:
                leading = (batch, time, 1)
            else:
                raise ValueError(
                    f"Feature {name!r} expected batch {batch * time * views} or {batch * time}, "
                    f"got {feature.shape[0]}."
                )
            return feature.reshape(*leading, *feature.shape[1:]).permute(0, 1, 2, 4, 5, 3).reshape(
                batch,
                time,
                -1,
                self.token_dim,
            )

        if feature.ndim == 3:
            if feature.shape[:2] != (batch, time):
                raise ValueError(f"Feature {name!r} expected leading [B,T], got {tuple(feature.shape[:2])}.")
            self._check_channels(name, feature.shape[2])
            return feature.unsqueeze(2)

        if feature.ndim == 2:
            self._check_channels(name, feature.shape[1])
            if feature.shape[0] == batch * time * views:
                return feature.view(batch, time, views, self.token_dim)
            if feature.shape[0] == batch * time:
                return feature.view(batch, time, 1, self.token_dim)
            raise ValueError(
                f"Feature {name!r} expected batch {batch * time * views} or {batch * time}, got {feature.shape[0]}."
            )

        raise ValueError(f"reshape_tokens expected feature {name!r} to be 2D-6D, got {tuple(feature.shape)}.")

    def _state_tokens(self, state: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        tokens = torch.zeros(
            state.shape[0],
            state.shape[1],
            1,
            self.token_dim,
            device=state.device,
            dtype=dtype,
        )
        tokens[..., : self.state_dim] = state.float().to(dtype=dtype).unsqueeze(2)
        return tokens

    def _feature_bias_from_logits(
        self,
        logits: torch.Tensor,
        *,
        batch: int,
        time: int,
        views: int,
        feature_hw: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        logits = logits.float()
        logits = logits - logits.mean(dim=(-2, -1), keepdim=True)
        upsampled = F.interpolate(logits, size=feature_hw, mode="bilinear", align_corners=False)
        flat = upsampled.flatten(2).squeeze(1).to(dtype=dtype) * self.saliency_bias_strength
        return flat.view(batch, time, views * feature_hw[0] * feature_hw[1])

    def _saliency_logits(
        self,
        backbone_out: dict[str, torch.Tensor],
        *,
        batch: int,
        time: int,
        views: int,
    ) -> torch.Tensor:
        if self.saliency_head is None:
            raise RuntimeError("Saliency is enabled but saliency_head is not initialized.")
        source = backbone_out[self.saliency_source_feature]
        if source.ndim != 4 or source.shape[0] != batch * time * views:
            raise ValueError(
                "adapter.saliency expects source_feature shape [B*T*V,C,H,W], "
                f"got {tuple(source.shape)}."
            )
        return self.saliency_head(source, self.saliency_grid_size)

    def _saliency_aux_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        sparsity = (probs.mean(dim=(-2, -1)) - self.saliency_keep_ratio).pow(2).mean()
        eps = 1.0e-6
        entropy = -(
            probs * torch.log(probs.clamp_min(eps))
            + (1 - probs) * torch.log((1 - probs).clamp_min(eps))
        ).mean()
        return (
            self.saliency_sparsity_loss_weight * sparsity
            + self.saliency_entropy_loss_weight * entropy
        )

    def _condition_bias(
        self,
        backbone_out: dict[str, torch.Tensor],
        *,
        batch: int,
        time: int,
        views: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.saliency_enabled:
            return None, None
        source = backbone_out[self.saliency_source_feature]
        logits = self._saliency_logits(backbone_out, batch=batch, time=time, views=views)
        source_bias = self._feature_bias_from_logits(
            logits,
            batch=batch,
            time=time,
            views=views,
            feature_hw=self.feature_hw[self.saliency_source_feature],
            dtype=dtype,
        )
        frame_bias_parts = []
        for name in self.feature_names:
            token_count = self.feature_view_counts[name] * self.feature_token_counts[name]
            if name == self.saliency_source_feature:
                frame_bias_parts.append(source_bias)
            else:
                frame_bias_parts.append(
                    torch.zeros(batch, time, token_count, device=source.device, dtype=dtype)
                )
        state_bias = torch.zeros(batch, time, 1, device=source.device, dtype=dtype)
        condition_bias = torch.cat([*frame_bias_parts, state_bias], dim=2).flatten(1, 2)
        if condition_bias.shape[1] != self.condition_len:
            raise ValueError(
                f"saliency inferred condition_len={self.condition_len}, "
                f"but produced {condition_bias.shape[1]} bias values."
            )
        aux_loss = self._saliency_aux_loss(logits) if self.training else None
        return condition_bias, aux_loss

    def _source_positions(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        height, width = self.feature_hw[self.saliency_source_feature]
        view_count = self.feature_view_counts[self.saliency_source_feature]
        ys = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        per_view = torch.stack([yy, xx], dim=-1).reshape(height * width, 2)
        return per_view.repeat(view_count, 1)

    def _prune_tokens_with_saliency(
        self,
        tokens: torch.Tensor,
        backbone_out: dict[str, torch.Tensor],
        *,
        batch: int,
        time: int,
        views: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.token_pruning_enabled:
            raise RuntimeError("_prune_tokens_with_saliency called with token_pruning disabled.")
        logits = self._saliency_logits(backbone_out, batch=batch, time=time, views=views)
        source_bias = self._feature_bias_from_logits(
            logits,
            batch=batch,
            time=time,
            views=views,
            feature_hw=self.feature_hw[self.saliency_source_feature],
            dtype=tokens.dtype,
        )
        _, selected = torch.topk(source_bias.float(), k=self.pruned_visual_tokens_per_frame, dim=2)
        selected = selected.sort(dim=2).values
        token_index = selected.unsqueeze(-1).expand(-1, -1, -1, self.token_dim)
        pruned_tokens = tokens.gather(dim=2, index=token_index)
        pruned_bias = source_bias.gather(dim=2, index=selected)

        base_positions = self._source_positions(device=tokens.device, dtype=tokens.dtype)
        positions = base_positions.unsqueeze(0).unsqueeze(0).expand(batch, time, -1, -1)
        position_index = selected.unsqueeze(-1).expand(-1, -1, -1, 2)
        pruned_positions = positions.gather(dim=2, index=position_index)
        aux_loss = self._saliency_aux_loss(logits) if self.training else None
        return pruned_tokens, pruned_bias, pruned_positions, aux_loss

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
            raise ValueError("adapter.type='reshape_tokens' does not expose attention maps.")
        if time != self.num_history_frames:
            raise ValueError(f"Expected time={self.num_history_frames}, got {time}.")
        if views != self.num_stereo_pairs:
            raise ValueError(f"Expected views={self.num_stereo_pairs}, got {views}.")
        if state.shape[:2] != (batch, time) or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state shape [B,{time},{self.state_dim}], got {tuple(state.shape)}.")

        parts = [
            self._feature_to_tokens(name, backbone_out[name], batch=batch, time=time, views=views)
            for name in self.feature_names
        ]
        state_tokens = self._state_tokens(state, dtype=parts[0].dtype)
        token_positions = None
        if self.token_pruning_enabled:
            visual_tokens, condition_bias, visual_positions, aux_loss = self._prune_tokens_with_saliency(
                parts[0],
                backbone_out,
                batch=batch,
                time=time,
                views=views,
            )
            state_bias = torch.zeros(batch, time, 1, device=state.device, dtype=condition_bias.dtype)
            state_positions = torch.zeros(batch, time, 1, 2, device=state.device, dtype=visual_positions.dtype)
            tokens = torch.cat([visual_tokens, state_tokens], dim=2).flatten(1, 2)
            condition_bias = torch.cat([condition_bias, state_bias], dim=2).flatten(1, 2)
            token_positions = torch.cat([visual_positions, state_positions], dim=2).flatten(1, 2)
        else:
            tokens = torch.cat([*parts, state_tokens], dim=2).flatten(1, 2)
            condition_bias, aux_loss = self._condition_bias(
                backbone_out,
                batch=batch,
                time=time,
                views=views,
                dtype=tokens.dtype,
            )
        if tokens.shape[1] != self.condition_len:
            raise ValueError(
                f"reshape_tokens inferred condition_len={self.condition_len}, "
                f"but runtime features produced {tokens.shape[1]} tokens."
            )
        if self.flatten:
            return AdapterOutput(cond=tokens.flatten(start_dim=1))
        return AdapterOutput(
            tokens=tokens,
            condition_bias=condition_bias,
            token_positions=token_positions,
            aux_loss=aux_loss,
        )


__all__ = ["CoarseSaliencyHead", "ReshapeTokensAdapter"]
