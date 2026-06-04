from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ffs.backbones import FoundationStereoBackbone, WAFTStereoBackbone
from ffs.heads import ActionHead, build_action_head
from ffs.policies.robomimic_diffusion_policy import (
    build_robomimic_diffusion_policy,
    build_spatial_backbone_diffusion_policy,
)


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
        disparity_ablation: str = "none",
        condition_token_dim: int = 256,
        feature_queries_per_scale: int = 4,
        disp_queries: int = 8,
        spatial_query_num_heads: int = 8,
        disparity_fusion_cfg: dict[str, Any] | None = None,
        use_disparity_tokens: bool = True,
        max_disp: int = 192,
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
        self.disparity_ablation = disparity_ablation
        self.condition_token_dim = condition_token_dim
        self.feature_queries_per_scale = feature_queries_per_scale
        self.disp_queries = disp_queries
        self.max_disp = max_disp
        self.feature_names = tuple(backbone.feature_names)
        fusion_cfg = dict(disparity_fusion_cfg or {})
        self.disparity_fusion_enabled = bool(fusion_cfg.get("enabled", False))
        if self.disparity_fusion_enabled and not use_disparity:
            raise ValueError("disparity_fusion requires use_disparity=True.")
        self.use_disparity_tokens = bool(use_disparity and use_disparity_tokens)
        self.use_disparity = bool(use_disparity and (self.use_disparity_tokens or self.disparity_fusion_enabled))

        if condition_token_dim <= 0:
            raise ValueError("condition_token_dim must be positive.")
        if feature_queries_per_scale <= 0:
            raise ValueError("feature_queries_per_scale must be positive.")
        if disp_queries <= 0:
            raise ValueError("disp_queries must be positive.")
        if disparity_ablation not in {"none", "zero", "shuffle"}:
            raise ValueError("disparity_ablation must be one of: none, zero, shuffle.")

        feature_channels = getattr(backbone, "feature_channels", None)
        if not isinstance(feature_channels, dict):
            raise ValueError("Backbone must expose a feature_channels dict.")
        missing_channels = sorted(set(self.feature_names) - set(feature_channels))
        if missing_channels:
            raise ValueError(f"Backbone is missing feature channels for: {', '.join(missing_channels)}")

        self.disparity_fusion_feature_names: tuple[str, ...] = ()
        self.disparity_fusion_encoder: DisparityFusionEncoder | None = None
        disparity_fusion_channels = 0
        if self.disparity_fusion_enabled:
            configured_names = fusion_cfg.get("feature_names", self.feature_names)
            self.disparity_fusion_feature_names = tuple(configured_names)
            unknown_fusion_names = sorted(set(self.disparity_fusion_feature_names) - set(self.feature_names))
            if unknown_fusion_names:
                raise ValueError(
                    "disparity_fusion.feature_names must be selected backbone feature names; "
                    f"unknown: {', '.join(unknown_fusion_names)}"
                )
            self.disparity_fusion_encoder = DisparityFusionEncoder(
                max_disp=max_disp,
                input_mode=fusion_cfg.get("input_mode", "disp_xy"),
                hidden_channels=int(fusion_cfg.get("hidden_channels", 16)),
                output_channels=int(fusion_cfg.get("output_channels", 16)),
                num_layers=int(fusion_cfg.get("num_layers", 2)),
            )
            disparity_fusion_channels = self.disparity_fusion_encoder.output_channels

        self.feature_resamplers = nn.ModuleDict(
            {
                name: StateConditionedSpatialResampler(
                    in_channels=int(feature_channels[name])
                    + (disparity_fusion_channels if name in self.disparity_fusion_feature_names else 0),
                    state_dim=state_dim,
                    token_dim=condition_token_dim,
                    num_queries=feature_queries_per_scale,
                    num_heads=spatial_query_num_heads,
                )
                for name in self.feature_names
            }
        )
        self.state_proj = nn.Linear(state_dim, condition_token_dim)

        if self.use_disparity_tokens:
            self.disp_resampler = StateConditionedSpatialResampler(
                in_channels=1,
                state_dim=state_dim,
                token_dim=condition_token_dim,
                num_queries=disp_queries,
                num_heads=spatial_query_num_heads,
            )
        else:
            self.disp_resampler = None

        visual_tokens_per_pair = len(self.feature_names) * feature_queries_per_scale
        if self.use_disparity_tokens:
            visual_tokens_per_pair += disp_queries
        tokens_per_frame = num_stereo_pairs * visual_tokens_per_pair + 1
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
            num_history_frames=num_history_frames,
            tokens_per_frame=tokens_per_frame,
        )

    def _resample_visual_tokens(
        self,
        backbone_out: dict[str, torch.Tensor],
        state: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens = []
        attention_maps = {}
        for name in self.feature_names:
            resampled = self.feature_resamplers[name](
                backbone_out[name],
                state,
                return_attention=return_attention,
            )
            if return_attention:
                token, attention = resampled
                attention_maps[name] = attention
            else:
                token = resampled
            tokens.append(token)
        tokens_out = torch.cat(tokens, dim=1)
        if return_attention:
            return tokens_out, attention_maps
        return tokens_out

    def _ablate_disparity(self, disp: torch.Tensor) -> torch.Tensor:
        if self.disparity_ablation == "none":
            return disp
        if self.disparity_ablation == "zero":
            return torch.zeros_like(disp)
        if self.disparity_ablation == "shuffle":
            spatial_size = disp.shape[-2] * disp.shape[-1]
            if spatial_size <= 1:
                return disp
            stride = spatial_size // 2 + 1
            while math.gcd(stride, spatial_size) != 1:
                stride += 1
            offset = spatial_size // 3 + 1
            perm = (torch.arange(spatial_size, device=disp.device) * stride + offset) % spatial_size
            return disp.flatten(2).index_select(2, perm).view_as(disp)
        raise ValueError(f"Unsupported disparity_ablation: {self.disparity_ablation}")

    def _fuse_disparity_features(
        self,
        backbone_out: dict[str, torch.Tensor],
        disp: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if not self.disparity_fusion_enabled:
            return backbone_out
        if disp is None:
            raise ValueError("Backbone did not return disparity needed for disparity_fusion.")
        if self.disparity_fusion_encoder is None:
            raise RuntimeError("disparity_fusion is enabled but encoder was not initialized.")

        disp_features = self.disparity_fusion_encoder(disp)
        fused = dict(backbone_out)
        for name in self.disparity_fusion_feature_names:
            feature = fused[name]
            resized = F.interpolate(
                disp_features,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=feature.dtype)
            fused[name] = torch.cat([feature, resized], dim=1)
        return fused

    def encode_tokens(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, t, v, c, h, w = left.shape
        left = left.reshape(b * t * v, c, h, w)
        right = right.reshape(b * t * v, c, h, w)
        state_by_frame = state.reshape(b, t, self.state_dim)
        state_by_pair = state_by_frame.unsqueeze(2).expand(b, t, v, self.state_dim)
        state_by_pair = state_by_pair.reshape(b * t * v, self.state_dim)

        backbone_out = self.backbone(left, right)
        disp = None
        if self.use_disparity:
            disp = self._ablate_disparity(backbone_out["disp"]).float()
            backbone_out = self._fuse_disparity_features(backbone_out, disp)

        attention_maps = {}
        visual_out = self._resample_visual_tokens(
            backbone_out,
            state_by_pair,
            return_attention=return_attention,
        )
        if return_attention:
            visual_tokens, visual_attention = visual_out
            for name, attention in visual_attention.items():
                attention_maps[name] = attention.view(
                    b,
                    t,
                    v,
                    attention.shape[1],
                    attention.shape[2],
                    attention.shape[3],
                    attention.shape[4],
                )
        else:
            visual_tokens = visual_out
        visual_tokens = visual_tokens.view(
            b,
            t,
            v,
            len(self.feature_names) * self.feature_queries_per_scale,
            -1,
        ).flatten(2, 3)
        frame_parts = [visual_tokens]

        if self.use_disparity_tokens:
            # disp: [B*T*V, 1, H, W] -> query tokens: [B, T, V*disp_queries, C]
            if disp is None:
                raise ValueError("Backbone did not return disparity needed for disparity tokens.")
            disp_out = self.disp_resampler(
                disp,
                state_by_pair,
                return_attention=return_attention,
            )
            if return_attention:
                disp_tokens, disp_attention = disp_out
                attention_maps["disp"] = disp_attention.view(
                    b,
                    t,
                    v,
                    disp_attention.shape[1],
                    disp_attention.shape[2],
                    disp_attention.shape[3],
                    disp_attention.shape[4],
                )
            else:
                disp_tokens = disp_out
            disp_tokens = disp_tokens.view(b, t, v, self.disp_queries, -1).flatten(2, 3)
            frame_parts.append(disp_tokens)

        state_tokens = self.state_proj(state.reshape(b, t, self.state_dim)).unsqueeze(2)
        frame_parts.append(state_tokens)
        tokens = torch.cat(frame_parts, dim=2).flatten(1, 2)
        if return_attention:
            return tokens, attention_maps
        return tokens

    def training_loss(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_head.training_loss(self.encode_tokens(left, right, state), action)

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
        if action is not None:
            return self.training_loss(left, right, state, action)
        encoded = self.encode_tokens(left, right, state, return_attention=return_attention)
        if return_attention:
            tokens, attention = encoded
            return self.action_head(tokens), attention
        return self.action_head(encoded)


def resolve_action_head_cfg(head_cfg: dict[str, Any]) -> dict[str, Any]:
    head_type = head_cfg.get("type", "mlp")
    nested_cfg = head_cfg.get(head_type)
    if isinstance(nested_cfg, dict):
        return {"type": head_type, **nested_cfg}

    reserved_keys = {
        "condition_token_dim",
        "feature_queries_per_scale",
        "disp_queries",
        "spatial_query_num_heads",
        "disparity_fusion",
        "use_disparity_tokens",
        "mlp",
        "rdt",
    }
    return {k: v for k, v in head_cfg.items() if k not in reserved_keys}


def build_cnn_disparity_provider(backbone_cfg: dict[str, Any]) -> tuple[nn.Module | None, int | None]:
    disparity_cfg = backbone_cfg.get("disparity")
    if not isinstance(disparity_cfg, dict) or not bool(disparity_cfg.get("enabled", False)):
        return None, None

    source = str(disparity_cfg.get("source", "ffs"))
    if source not in {"ffs", "waft"}:
        raise ValueError("backbone.disparity.source must be one of: ffs, waft.")
    default_max_disp = 192 if source == "ffs" else 800
    max_disp = int(disparity_cfg.get("max_disp", default_max_disp))
    freeze = bool(disparity_cfg.get("freeze", True))

    if source == "ffs":
        ffs_cfg = dict(disparity_cfg.get("ffs") or {})
        provider = FoundationStereoBackbone(
            foundation_root=ffs_cfg["foundation_root"],
            checkpoint_path=ffs_cfg["checkpoint_path"],
            valid_iters=ffs_cfg.get("valid_iters", 4),
            max_disp=max_disp,
            freeze=freeze,
            use_disparity=True,
            optimize_build_volume=ffs_cfg.get("optimize_build_volume", "pytorch1"),
            feature_names=ffs_cfg.get("feature_names", ["feat_04"]),
        )
        return provider, max_disp

    waft_cfg = dict(disparity_cfg.get("waft") or {})
    provider = WAFTStereoBackbone(
        waft_root=waft_cfg["waft_root"],
        config_path=waft_cfg["config_path"],
        checkpoint_path=waft_cfg["checkpoint_path"],
        freeze=freeze,
        use_disparity=True,
        feature_names=waft_cfg.get("feature_names", ["fmap1"]),
        amp_dtype=waft_cfg.get("amp_dtype"),
    )
    return provider, max_disp


def build_policy(cfg: dict[str, Any]) -> StereoActionPolicy:
    backbone_cfg = cfg["backbone"]
    policy_cfg = cfg["policy"]
    head_cfg = cfg.get("head", {})
    backbone_type = backbone_cfg.get("type", "ffs-based")
    head_type = head_cfg.get("type", "mlp")

    if backbone_type == "cnn-based":
        if head_type != "diffusion_unet":
            raise ValueError(
                "backbone.type='cnn-based' currently requires head.type='diffusion_unet'. "
                f"Got head.type={head_type!r}."
            )
        disparity_provider, disparity_max_disp = build_cnn_disparity_provider(backbone_cfg)
        return build_robomimic_diffusion_policy(
            backbone_cfg=backbone_cfg,
            policy_cfg=policy_cfg,
            head_cfg=head_cfg,
            image_size=cfg.get("dataset", {}).get("image_size", [224, 224]),
            disparity_provider=disparity_provider,
            disparity_max_disp=disparity_max_disp,
        )

    if backbone_type not in {"ffs-based", "waft-based"}:
        raise ValueError("backbone.type must be one of: cnn-based, ffs-based, waft-based.")
    dataset_cfg = cfg.get("dataset", {})
    action_head_cfg = resolve_action_head_cfg(head_cfg)
    num_stereo_pairs = len(dataset_cfg.get("camera_pairs", []))
    policy_use_disparity = bool(policy_cfg.get("use_disparity", True))
    disparity_fusion_cfg = head_cfg.get("disparity_fusion")
    disparity_fusion_enabled = bool(
        isinstance(disparity_fusion_cfg, dict) and disparity_fusion_cfg.get("enabled", False)
    )
    use_disparity_tokens = bool(head_cfg.get("use_disparity_tokens", True))
    if head_type == "diffusion_unet":
        backbone_use_disparity = bool(backbone_cfg.get("use_disparity", head_cfg.get("use_disparity", False)))
    else:
        backbone_use_disparity = policy_use_disparity and (use_disparity_tokens or disparity_fusion_enabled)

    if backbone_type == "ffs-based":
        backbone = FoundationStereoBackbone(
            foundation_root=backbone_cfg["foundation_root"],
            checkpoint_path=backbone_cfg["checkpoint_path"],
            valid_iters=backbone_cfg.get("valid_iters", 4),
            max_disp=backbone_cfg.get("max_disp", 192),
            freeze=backbone_cfg.get("freeze", True),
            use_disparity=backbone_use_disparity,
            optimize_build_volume=backbone_cfg.get("optimize_build_volume", "pytorch1"),
            feature_names=backbone_cfg["feature_names"],
        )
    else:
        backbone = WAFTStereoBackbone(
            waft_root=backbone_cfg["waft_root"],
            config_path=backbone_cfg["config_path"],
            checkpoint_path=backbone_cfg["checkpoint_path"],
            freeze=backbone_cfg.get("freeze", True),
            use_disparity=backbone_use_disparity,
            feature_names=backbone_cfg["feature_names"],
            amp_dtype=backbone_cfg.get("amp_dtype"),
        )
    max_disp = int(backbone_cfg.get("max_disp", getattr(backbone, "max_disp", 192)))

    if head_type == "diffusion_unet":
        return build_spatial_backbone_diffusion_policy(
            backbone=backbone,
            policy_cfg=policy_cfg,
            head_cfg=head_cfg,
            num_stereo_pairs=num_stereo_pairs,
        )

    return StereoActionPolicy(
        backbone=backbone,
        num_history_frames=policy_cfg["num_history_frames"],
        state_dim=policy_cfg["state_dim"],
        action_dim=policy_cfg["action_dim"],
        action_horizon=policy_cfg["action_horizon"],
        num_stereo_pairs=num_stereo_pairs,
        use_disparity=backbone_use_disparity,
        disparity_ablation=policy_cfg.get("disparity_ablation", "none"),
        condition_token_dim=head_cfg["condition_token_dim"],
        feature_queries_per_scale=head_cfg.get("feature_queries_per_scale", 4),
        disp_queries=head_cfg.get("disp_queries", 8),
        spatial_query_num_heads=head_cfg.get("spatial_query_num_heads", 8),
        disparity_fusion_cfg=disparity_fusion_cfg,
        use_disparity_tokens=use_disparity_tokens,
        max_disp=max_disp,
        action_head_cfg=action_head_cfg,
    )
