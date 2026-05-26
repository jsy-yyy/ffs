from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn


class FoundationStereoBackbone(nn.Module):
    """Frozen Fast-FoundationStereo wrapper.

    Inputs are RGB tensors in 0..255 range with shape [N, 3, H, W].
    """

    all_feature_names = ("feat_04", "feat_08", "feat_16", "feat_32")
    feature_names = all_feature_names

    def __init__(
        self,
        foundation_root: str | Path,
        checkpoint_path: str | Path,
        valid_iters: int = 4,
        max_disp: int = 192,
        freeze: bool = True,
        use_disparity: bool = True,
        optimize_build_volume: str = "pytorch1",
        feature_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.foundation_root = Path(foundation_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.valid_iters = valid_iters
        self.max_disp = max_disp
        self.freeze = freeze
        self.use_disparity = use_disparity
        self.optimize_build_volume = optimize_build_volume
        self.feature_names = self._validate_feature_names(feature_names)

        self._add_foundation_root()
        self._patch_timm_layers_alias()
        importlib.import_module("core.foundation_stereo")
        self.model = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model.args.valid_iters = valid_iters
        self.model.args.max_disp = max_disp

        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

        self.feature_channels = dict(zip(self.all_feature_names, self.model.feature.d_out))
        self.visual_token_dim = sum(self.feature_channels[name] for name in self.feature_names)

    def _validate_feature_names(self, feature_names: Sequence[str] | None) -> tuple[str, ...]:
        if feature_names is None:
            return self.all_feature_names
        selected = tuple(feature_names)
        if not selected:
            raise ValueError("backbone.feature_names must be a non-empty list.")
        unknown = sorted(set(selected) - set(self.all_feature_names))
        if unknown:
            available = ", ".join(self.all_feature_names)
            raise ValueError(
                f"Unknown backbone feature names: {', '.join(unknown)}. Available: {available}"
            )
        return selected

    def _add_foundation_root(self) -> None:
        root = str(self.foundation_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _patch_timm_layers_alias(self) -> None:
        try:
            importlib.import_module("timm.layers")
        except ModuleNotFoundError:
            sys.modules["timm.layers"] = importlib.import_module("timm.models.layers")

    def _sync_model_device(self, device: torch.device) -> None:
        model_device = next(self.model.parameters()).device
        if model_device != device:
            self.model.to(device)

    def extract_features(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        from core.foundation_stereo import normalize_image

        self._sync_model_device(left.device)
        b = left.shape[0]
        left = normalize_image(left)
        right = normalize_image(right)
        features = self.model.feature(torch.cat([left, right], dim=0))
        all_features = {name: feat[:b] for name, feat in zip(self.all_feature_names, features)}
        return {name: all_features[name] for name in self.feature_names}

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.freeze:
            self.model.eval()
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            out = self.extract_features(left, right)
            if self.use_disparity:
                self.model.args.mixed_precision = bool(left.is_cuda and self.model.args.mixed_precision)
                out["disp"] = self.model.forward(
                    left,
                    right,
                    iters=self.valid_iters,
                    test_mode=True,
                    optimize_build_volume=self.optimize_build_volume,
                )
        return out
