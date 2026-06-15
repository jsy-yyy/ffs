from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoDenseFeatureBranch(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        backend: str = "huggingface",
        torchhub_model: str = "dinov2_vitb14",
        repo_or_dir: str = "facebookresearch/dinov2",
        source: str = "github",
        checkpoint_path: str | None = None,
        output_channels: int = 128,
        local_files_only: bool = True,
        freeze: bool = True,
        projection: str = "conv_downsample",
    ) -> None:
        super().__init__()
        self.backend = str(backend)
        if self.backend not in {"huggingface", "torchhub"}:
            raise ValueError("dino.backend must be one of: huggingface, torchhub.")
        self.projection = str(projection)
        if self.projection not in {"conv_downsample", "linear"}:
            raise ValueError("dino.projection must be one of: conv_downsample, linear.")

        if self.backend == "huggingface":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
            hidden_size = int(self.model.config.hidden_size)
            self.patch_size = int(getattr(self.model.config, "patch_size", 14))
        else:
            hub_kwargs: dict[str, Any] = {}
            if checkpoint_path is not None:
                hub_kwargs["weights"] = checkpoint_path
            self.model = torch.hub.load(
                repo_or_dir,
                torchhub_model,
                pretrained=True,
                source=source,
                **hub_kwargs,
            )
            hidden_size = int(getattr(self.model, "embed_dim"))
            patch_size = getattr(getattr(self.model, "patch_embed"), "patch_size", (14, 14))
            self.patch_size = int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)

        if self.projection == "conv_downsample":
            self.projector = nn.Conv2d(hidden_size, output_channels, kernel_size=4, stride=4)
        else:
            self.projector = nn.Linear(hidden_size, output_channels)

        self.output_channels = int(output_channels)
        self.freeze = bool(freeze)
        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() / 255.0).clamp(0.0, 1.0)
        return (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)

    def _grid_size(self, image_height: int, image_width: int, num_tokens: int) -> tuple[int, int]:
        height = max(image_height // self.patch_size, 1)
        width = max(image_width // self.patch_size, 1)
        if height * width == num_tokens:
            return height, width
        side = int(num_tokens**0.5)
        if side * side == num_tokens:
            return side, side
        raise ValueError(f"Cannot reshape DINO patch tokens: num_tokens={num_tokens}, image={image_height}x{image_width}.")

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = self._normalize(x)
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            if self.backend == "huggingface":
                try:
                    output = self.model(pixel_values=pixel_values, interpolate_pos_encoding=True)
                except TypeError:
                    output = self.model(pixel_values=pixel_values)
                return output.last_hidden_state[:, 1:]

            if not hasattr(self.model, "forward_features"):
                raise ValueError("torchhub DINOv2 model must expose forward_features.")
            output = self.model.forward_features(pixel_values)
            if isinstance(output, dict) and "x_norm_patchtokens" in output:
                return output["x_norm_patchtokens"]
            if isinstance(output, dict) and "x_prenorm" in output:
                num_prefix = 1 + int(getattr(self.model, "num_register_tokens", 0))
                return output["x_prenorm"][:, num_prefix:]
            raise ValueError("Unsupported torchhub DINOv2 forward_features output.")

    def forward(self, x: torch.Tensor, output_size: tuple[int, int] | None) -> torch.Tensor:
        if self.freeze:
            self.model.eval()
        tokens = self._patch_tokens(x)
        grid_h, grid_w = self._grid_size(x.shape[-2], x.shape[-1], tokens.shape[1])

        if self.projection == "linear":
            features = self.projector(tokens).transpose(1, 2).reshape(
                x.shape[0],
                self.output_channels,
                grid_h,
                grid_w,
            )
        else:
            features = tokens.transpose(1, 2).reshape(x.shape[0], tokens.shape[2], grid_h, grid_w)
            if features.shape[-2] < 4 or features.shape[-1] < 4:
                pad_h = max(4 - features.shape[-2], 0)
                pad_w = max(4 - features.shape[-1], 0)
                features = F.pad(features, (0, pad_w, 0, pad_h))
            features = self.projector(features)

        if output_size is None:
            return features
        return F.interpolate(features, size=output_size, mode="bilinear", align_corners=False)


__all__ = ["DinoDenseFeatureBranch"]
