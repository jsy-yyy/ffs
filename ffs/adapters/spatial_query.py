from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import AdapterOutput, BaseAdapter, resolve_feature_names


def _sincos_1d_pos_embed(dim: int, pos: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return torch.zeros(pos.numel(), dim, device=pos.device, dtype=torch.float32)
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=pos.device))
        * torch.arange(half, device=pos.device, dtype=torch.float32)
        / half
    )
    args = pos.reshape(-1).float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = torch.cat(
            [emb, torch.zeros(emb.shape[0], dim - emb.shape[1], device=pos.device)],
            dim=1,
        )
    return emb


def _sincos_2d_pos_embed(dim: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    y_dim = dim // 2
    x_dim = dim - y_dim
    return torch.cat(
        [
            _sincos_1d_pos_embed(y_dim, y.flatten()),
            _sincos_1d_pos_embed(x_dim, x.flatten()),
        ],
        dim=1,
    )


class StateConditionedSpatialResampler(nn.Module):
    """Read a fixed number of state-conditioned tokens from a spatial map."""

    def __init__(
        self,
        in_channels: int,
        state_dim: int,
        token_dim: int,
        num_queries: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")

        self.token_dim = token_dim
        self.num_queries = num_queries
        self.input_proj = nn.Conv2d(in_channels, token_dim, kernel_size=1)
        self.spatial_norm = nn.LayerNorm(token_dim)
        self.query_norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(token_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, token_dim) * 0.02)
        self.state_modulation = nn.Linear(state_dim, token_dim * 2)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = x.shape
        spatial = self.input_proj(x).flatten(2).transpose(1, 2)
        pos = _sincos_2d_pos_embed(self.token_dim, height, width, x.device).to(dtype=spatial.dtype)
        spatial = self.spatial_norm(spatial + pos.unsqueeze(0))

        shift, scale = self.state_modulation(state).chunk(2, dim=1)
        queries = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        queries = queries * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        queries = self.query_norm(queries)

        if return_attention:
            tokens, attention = self.attn(
                queries,
                spatial,
                spatial,
                need_weights=True,
                average_attn_weights=False,
            )
            attention = attention.view(x.shape[0], -1, self.num_queries, height, width)
            return self.out_norm(tokens), attention

        tokens = self.attn(queries, spatial, spatial, need_weights=False)[0]
        return self.out_norm(tokens)


class DisparityFusionEncoder(nn.Module):
    """Encode dense disparity into local geometry features for multi-scale fusion."""

    def __init__(
        self,
        max_disp: int = 192,
        input_mode: str = "disp_xy",
        hidden_channels: int = 16,
        output_channels: int = 16,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if max_disp <= 0:
            raise ValueError("max_disp must be positive.")
        if input_mode not in {"disp", "disp_xy"}:
            raise ValueError("disparity fusion input_mode must be one of: disp, disp_xy.")
        if hidden_channels <= 0:
            raise ValueError("disparity fusion hidden_channels must be positive.")
        if output_channels <= 0:
            raise ValueError("disparity fusion output_channels must be positive.")
        if num_layers <= 0:
            raise ValueError("disparity fusion num_layers must be positive.")

        self.max_disp = float(max_disp)
        self.input_mode = input_mode
        self.output_channels = output_channels

        in_channels = 1 if input_mode == "disp" else 3
        layers: list[nn.Module] = []
        for layer_idx in range(num_layers):
            out_channels = output_channels if layer_idx == num_layers - 1 else hidden_channels
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.SiLU())
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, disp: torch.Tensor) -> torch.Tensor:
        disp = (disp.float() / self.max_disp).clamp(0.0, 1.0)
        if self.input_mode == "disp_xy":
            _, _, height, width = disp.shape
            y = torch.linspace(0.0, 1.0, height, device=disp.device, dtype=disp.dtype)
            x = torch.linspace(0.0, 1.0, width, device=disp.device, dtype=disp.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            xy = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(disp.shape[0], -1, -1, -1)
            disp = torch.cat([disp, xy], dim=1)
        return self.net(disp)


class SpatialQueryAdapter(BaseAdapter):
    output_kind = "tokens"

    def __init__(
        self,
        *,
        backbone: nn.Module,
        policy_cfg: dict[str, Any],
        dataset_cfg: dict[str, Any],
        feature_names: object = "auto",
        token_dim: int = 256,
        queries_per_feature: int = 4,
        num_heads: int = 8,
        flatten: bool = False,
    ) -> None:
        super().__init__()
        self.feature_names = resolve_feature_names(feature_names, backbone)
        self.num_history_frames = int(policy_cfg.get("observation_horizon", policy_cfg["num_history_frames"]))
        self.state_dim = int(policy_cfg["state_dim"])
        self.num_stereo_pairs = len(dataset_cfg.get("camera_pairs", []))
        self.token_dim = int(token_dim)
        self.queries_per_feature = int(queries_per_feature)
        self.flatten = bool(flatten)
        self.max_disp = int(getattr(backbone, "max_disp", 192))

        feature_channels = getattr(backbone, "feature_channels", None)
        if not isinstance(feature_channels, dict):
            raise ValueError("Backbone must expose a feature_channels dict.")
        missing = sorted(set(self.feature_names) - set(feature_channels))
        if missing:
            raise ValueError(f"Backbone is missing feature channels for: {', '.join(missing)}")

        self.feature_resamplers = nn.ModuleDict(
            {
                name: StateConditionedSpatialResampler(
                    in_channels=int(feature_channels[name]),
                    state_dim=self.state_dim,
                    token_dim=self.token_dim,
                    num_queries=self.queries_per_feature,
                    num_heads=int(num_heads),
                )
                for name in self.feature_names
            }
        )
        self.state_proj = nn.Linear(self.state_dim, self.token_dim)

        visual_tokens_per_pair = len(self.feature_names) * self.queries_per_feature
        self.tokens_per_frame = self.num_stereo_pairs * visual_tokens_per_pair + 1
        self.condition_len = self.num_history_frames * self.tokens_per_frame
        self.cond_dim = self.condition_len * self.token_dim
        self.output_kind = "cond" if self.flatten else "tokens"

    def _prepare_feature(self, name: str, feature: torch.Tensor) -> torch.Tensor:
        if name != "disp":
            return feature
        if self.max_disp <= 0:
            raise ValueError("Backbone must expose a positive max_disp when using 'disp' as a feature.")
        return (feature.float() / self.max_disp).clamp(0.0, 1.0).to(dtype=feature.dtype)

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
        if time != self.num_history_frames:
            raise ValueError(f"Expected time={self.num_history_frames}, got {time}.")
        if views != self.num_stereo_pairs:
            raise ValueError(f"Expected views={self.num_stereo_pairs}, got {views}.")
        if state.shape[:2] != (batch, time) or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state shape [B,{time},{self.state_dim}], got {tuple(state.shape)}.")

        state_by_pair = state.reshape(batch, time, self.state_dim).unsqueeze(2).expand(
            batch,
            time,
            views,
            self.state_dim,
        )
        state_by_pair = state_by_pair.reshape(batch * time * views, self.state_dim)

        attention_maps: dict[str, torch.Tensor] = {}
        tokens = []
        for name in self.feature_names:
            feature = self._prepare_feature(name, backbone_out[name])
            if feature.ndim != 4 or feature.shape[0] != batch * time * views:
                raise ValueError(
                    f"spatial_query expects feature {name!r} shape [B*T*V,C,H,W], got {tuple(feature.shape)}."
                )
            resampled = self.feature_resamplers[name](feature, state_by_pair, return_attention=return_attention)
            if return_attention:
                token, attention = resampled
                attention_maps[name] = attention.view(
                    batch,
                    time,
                    views,
                    attention.shape[1],
                    attention.shape[2],
                    attention.shape[3],
                    attention.shape[4],
                )
            else:
                token = resampled
            tokens.append(token)

        visual_tokens = torch.cat(tokens, dim=1).view(
            batch,
            time,
            views,
            len(self.feature_names) * self.queries_per_feature,
            -1,
        ).flatten(2, 3)
        frame_parts = [visual_tokens]

        state_tokens = self.state_proj(state.reshape(batch, time, self.state_dim)).unsqueeze(2)
        frame_parts.append(state_tokens)
        tokens_out = torch.cat(frame_parts, dim=2).flatten(1, 2)
        if self.flatten:
            return AdapterOutput(
                cond=tokens_out.flatten(start_dim=1),
                attention=attention_maps if return_attention else None,
            )
        return AdapterOutput(tokens=tokens_out, attention=attention_maps if return_attention else None)


__all__ = [
    "DisparityFusionEncoder",
    "SpatialQueryAdapter",
    "StateConditionedSpatialResampler",
]
