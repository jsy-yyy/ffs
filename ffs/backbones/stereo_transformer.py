from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from torchvision import models as vision_models
from torchvision.ops import FeaturePyramidNetwork

from .dino import DinoDenseFeatureBranch


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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_1d_rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim < 2:
        return x
    rope_dim = dim if dim % 2 == 0 else dim - 1
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    half = rope_dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=x.device))
        * torch.arange(half, device=x.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = pos.float().unsqueeze(1) * freqs.unsqueeze(0)
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=1).to(dtype=x.dtype)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=1).to(dtype=x.dtype)
    x_rope = x_rope * cos.view(1, 1, pos.numel(), rope_dim) + _rotate_half(x_rope) * sin.view(
        1,
        1,
        pos.numel(),
        rope_dim,
    )
    if x_pass.numel() == 0:
        return x_rope
    return torch.cat([x_rope, x_pass], dim=-1)


def _apply_2d_rope(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Apply 2D RoPE to [B, heads, H*W, head_dim] tensors."""

    _, _, tokens, dim = x.shape
    if tokens != height * width:
        raise ValueError(f"RoPE expected {height * width} tokens, got {tokens}.")
    y, x_pos = torch.meshgrid(
        torch.arange(height, device=x.device),
        torch.arange(width, device=x.device),
        indexing="ij",
    )
    y = y.flatten()
    x_pos = x_pos.flatten()
    y_dim = dim // 2
    x_dim = dim - y_dim
    return torch.cat(
        [
            _apply_1d_rope(x[..., :y_dim], y),
            _apply_1d_rope(x[..., y_dim:], x_pos),
        ],
        dim=-1,
    )


class ResNet18FPNEncoder(nn.Module):
    def __init__(self, out_channels: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        weights = vision_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = vision_models.resnet18(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], out_channels=out_channels)
        for unused_output_block in self.fpn.layer_blocks[1:]:
            unused_output_block.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() / 255.0).clamp(0.0, 1.0)
        return (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn({"c2": c2, "c3": c3, "c4": c4, "c5": c5})["c2"]


class TokenMLP(nn.Module):
    def __init__(self, in_channels: int, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention2d(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("token dim must be divisible by num_heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, context: torch.Tensor, height: int, width: int) -> torch.Tensor:
        q = _apply_2d_rope(self._split_heads(self.q_proj(query)), height, width)
        k = _apply_2d_rope(self._split_heads(self.k_proj(context)), height, width)
        v = self._split_heads(self.v_proj(context))
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(query.shape[0], query.shape[1], self.dim)
        return self.out_proj(out)


class StereoTransformerLayer(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(token_dim)
        self.self_attn = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(token_dim)
        self.cross_attn = CrossAttention2d(token_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(token_dim)
        hidden_dim = int(token_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_self = self.self_norm(left)
        right_self = self.self_norm(right)
        left = left + self.self_attn(left_self, left_self, left_self, need_weights=False)[0]
        right = right + self.self_attn(right_self, right_self, right_self, need_weights=False)[0]

        left_cross = self.cross_norm(left)
        right_cross = self.cross_norm(right)
        left = left + self.cross_attn(left_cross, right_cross, height, width)
        right = right + self.cross_attn(right_cross, left_cross, height, width)

        left = left + self.mlp(self.mlp_norm(left))
        right = right + self.mlp(self.mlp_norm(right))
        return left, right


class StereoTransformerBackbone(nn.Module):
    """Shared 2D encoder + stereo transformer producing one latent per camera pair.

    Inputs are RGB tensors in 0..255 range with shape [N, 3, H, W].
    """

    all_feature_names = ("stereo_latent",)
    feature_names = all_feature_names
    uses_view_indices = True

    def __init__(
        self,
        feature_names: list[str] | tuple[str, ...] | None = None,
        cnn_feature_dim: int = 128,
        token_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pretrained_resnet: bool = False,
        freeze_cnn: bool = False,
        dino: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.feature_names = self._validate_feature_names(feature_names)
        self.cnn_feature_dim = int(cnn_feature_dim)
        self.token_dim = int(token_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        if self.token_dim % self.num_heads != 0:
            raise ValueError("backbone.token_dim must be divisible by backbone.num_heads.")

        self.cnn = ResNet18FPNEncoder(out_channels=self.cnn_feature_dim, pretrained=pretrained_resnet)
        if freeze_cnn:
            self.cnn.requires_grad_(False)
            self.cnn.eval()
        self.freeze_cnn = bool(freeze_cnn)

        dino_cfg = dict(dino or {})
        self.dino_view_indices = tuple(int(idx) for idx in dino_cfg.get("view_indices", []))
        self.dino_feature_dim = int(dino_cfg.get("output_channels", 128))
        self.dino: DinoDenseFeatureBranch | None = None
        if self.dino_view_indices:
            dino_backend = dino_cfg.get("backend")
            if dino_backend is None and (
                dino_cfg.get("torchhub_model") is not None or dino_cfg.get("checkpoint_path") is not None
            ):
                dino_backend = "torchhub"
            self.dino = DinoDenseFeatureBranch(
                model_name=dino_cfg.get("model_name", "facebook/dinov2-base"),
                backend=dino_backend or "huggingface",
                torchhub_model=dino_cfg.get("torchhub_model", "dinov2_vitb14"),
                repo_or_dir=dino_cfg.get("repo_or_dir", "facebookresearch/dinov2"),
                source=dino_cfg.get("source", "github"),
                checkpoint_path=dino_cfg.get("checkpoint_path"),
                output_channels=self.dino_feature_dim,
                local_files_only=bool(dino_cfg.get("local_files_only", True)),
                freeze=bool(dino_cfg.get("freeze", True)),
            )

        projector_in_dim = self.cnn_feature_dim + (self.dino_feature_dim if self.dino is not None else 0)
        self.token_projector = TokenMLP(projector_in_dim, self.token_dim)
        self.layers = nn.ModuleList(
            [
                StereoTransformerLayer(
                    token_dim=self.token_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.token_dim)
        self.latent_query = nn.Parameter(torch.randn(1, self.token_dim) * 0.02)
        self.latent_pool = nn.MultiheadAttention(self.token_dim, self.num_heads, batch_first=True)
        self.latent_proj = nn.Linear(self.token_dim, self.latent_dim)
        self.feature_channels = {"stereo_latent": self.latent_dim}
        self.visual_token_dim = self.latent_dim

    def _validate_feature_names(self, feature_names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        if feature_names is None:
            return self.all_feature_names
        selected = tuple(feature_names)
        if not selected:
            raise ValueError("backbone.feature_names must be a non-empty list.")
        unknown = sorted(set(selected) - set(self.all_feature_names))
        if unknown:
            available = ", ".join(self.all_feature_names)
            raise ValueError(f"Unknown stereo-transformer feature names: {', '.join(unknown)}. Available: {available}")
        return selected

    def _dino_mask(self, view_indices: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.dino is None:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        if view_indices is None:
            return torch.ones(batch_size, device=device, dtype=torch.bool)
        mask = torch.zeros_like(view_indices, dtype=torch.bool, device=device)
        for view_idx in self.dino_view_indices:
            mask = mask | (view_indices.to(device=device) == view_idx)
        return mask

    def _encode_cnn(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_cnn:
            self.cnn.eval()
        ctx = torch.no_grad() if self.freeze_cnn else nullcontext()
        with ctx:
            return self.cnn(x)

    def _append_dino(
        self,
        features: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.dino is None:
            return features
        dino_features = torch.zeros(
            features.shape[0],
            self.dino_feature_dim,
            features.shape[-2],
            features.shape[-1],
            device=features.device,
            dtype=features.dtype,
        )
        if mask.any():
            selected = self.dino(image[mask], output_size=features.shape[-2:]).to(dtype=features.dtype)
            dino_features[mask] = selected
        return torch.cat([features, dino_features], dim=1)

    def _features_to_tokens(self, features: torch.Tensor) -> torch.Tensor:
        tokens = features.flatten(2).transpose(1, 2)
        return self.token_projector(tokens)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        view_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if left.shape != right.shape:
            raise ValueError(f"left/right image shapes must match, got {tuple(left.shape)} and {tuple(right.shape)}.")
        if left.ndim != 4 or left.shape[1] != 3:
            raise ValueError(f"StereoTransformerBackbone expects [N,3,H,W] images, got {tuple(left.shape)}.")
        if view_indices is not None and view_indices.shape[0] != left.shape[0]:
            raise ValueError(
                f"view_indices must have length {left.shape[0]}, got shape {tuple(view_indices.shape)}."
            )

        left_features = self._encode_cnn(left)
        right_features = self._encode_cnn(right)
        height, width = left_features.shape[-2:]
        mask = self._dino_mask(view_indices, left.shape[0], left.device)
        left_features = self._append_dino(left_features, left, mask)
        right_features = self._append_dino(right_features, right, mask)

        left_tokens = self._features_to_tokens(left_features)
        right_tokens = self._features_to_tokens(right_features)
        pos = _sincos_2d_pos_embed(self.token_dim, height, width, left.device).to(dtype=left_tokens.dtype)
        left_tokens = left_tokens + pos.unsqueeze(0)
        right_tokens = right_tokens + pos.unsqueeze(0)

        for layer in self.layers:
            left_tokens, right_tokens = layer(left_tokens, right_tokens, height, width)

        stereo_tokens = self.final_norm(torch.cat([left_tokens, right_tokens], dim=1))
        query = self.latent_query.to(dtype=stereo_tokens.dtype).unsqueeze(0).expand(left.shape[0], -1, -1)
        latent = self.latent_pool(query, stereo_tokens, stereo_tokens, need_weights=False)[0].squeeze(1)
        latent = self.latent_proj(latent)
        return {"stereo_latent": latent[:, :, None, None]}


__all__ = ["StereoTransformerBackbone"]
