from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

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
        self.tokens_per_frame = (
            sum(
                feature_view_count(backbone, name, self.num_stereo_pairs) * self.feature_token_counts[name]
                for name in self.feature_names
            )
            + 1
        )
        self.condition_len = self.num_history_frames * self.tokens_per_frame
        self.cond_dim = self.condition_len * self.token_dim
        self.flatten = bool(flatten)
        self.output_kind = "cond" if self.flatten else "tokens"

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
        tokens = torch.cat([*parts, state_tokens], dim=2).flatten(1, 2)
        if tokens.shape[1] != self.condition_len:
            raise ValueError(
                f"reshape_tokens inferred condition_len={self.condition_len}, "
                f"but runtime features produced {tokens.shape[1]} tokens."
            )
        if self.flatten:
            return AdapterOutput(cond=tokens.flatten(start_dim=1))
        return AdapterOutput(tokens=tokens)


__all__ = ["ReshapeTokensAdapter"]
