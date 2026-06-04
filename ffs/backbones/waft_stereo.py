from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import yaml


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_namespace(item) for item in value)
    return value


class WAFTStereoBackbone(nn.Module):
    """WAFT-Stereo wrapper that exposes spatial stereo features.

    Inputs are RGB tensors in 0..255 range with shape [N, 3, H, W].
    """

    all_feature_names = ("fmap1", "net", "disp")
    default_feature_names = ("fmap1", "net")

    def __init__(
        self,
        waft_root: str | Path,
        config_path: str | Path,
        checkpoint_path: str | Path,
        freeze: bool = True,
        use_disparity: bool = True,
        feature_names: Sequence[str] | None = None,
        amp_dtype: str | None = None,
    ) -> None:
        super().__init__()
        self.waft_root = Path(waft_root)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.freeze = bool(freeze)
        self.use_disparity = bool(use_disparity)
        self.feature_names = self._validate_feature_names(feature_names)
        if "disp" in self.feature_names and not self.use_disparity:
            raise ValueError("Selecting WAFT feature 'disp' requires backbone.use_disparity=True.")
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)

        self._add_waft_root()
        self._patch_import_compat()
        waft_module = importlib.import_module("algorithms.waft")
        cfg = self._load_config()
        self.model = waft_module.WAFT(cfg)
        self._load_checkpoint()

        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

        enc_dim = int(self.model.enc_dim)
        self.max_disp = int(getattr(self.model, "max_disp", 0))
        self.feature_channels = {"fmap1": enc_dim, "net": enc_dim, "disp": 1}
        self.visual_token_dim = sum(self.feature_channels[name] for name in self.feature_names)

    def _validate_feature_names(self, feature_names: Sequence[str] | None) -> tuple[str, ...]:
        if feature_names is None:
            return self.default_feature_names
        selected = tuple(feature_names)
        if not selected:
            raise ValueError("backbone.feature_names must be a non-empty list.")
        unknown = sorted(set(selected) - set(self.all_feature_names))
        if unknown:
            available = ", ".join(self.all_feature_names)
            raise ValueError(
                f"Unknown WAFT feature names: {', '.join(unknown)}. Available: {available}"
            )
        return selected

    @staticmethod
    def _resolve_amp_dtype(name: str | None) -> torch.dtype | None:
        if name is None:
            return None
        normalized = str(name).lower()
        if normalized in {"none", "false", "off"}:
            return None
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        raise ValueError("backbone.amp_dtype must be one of: bfloat16, float16, none.")

    def _add_waft_root(self) -> None:
        root = str(self.waft_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _patch_import_compat(self) -> None:
        try:
            importlib.import_module("timm.layers")
        except ModuleNotFoundError:
            sys.modules["timm.layers"] = importlib.import_module("timm.models.layers")

        try:
            import packaging
            import pkg_resources
        except ModuleNotFoundError:
            return
        if not hasattr(pkg_resources, "packaging"):
            pkg_resources.packaging = packaging

    def _load_config(self) -> SimpleNamespace:
        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"WAFT config must be a mapping, got {type(data).__name__}.")
        self._fill_config_defaults(data)
        return _namespace(data)

    def _fill_config_defaults(self, data: dict[str, Any]) -> None:
        waft = data.setdefault("WAFT", {})
        feature_encoder = waft.setdefault("FEATURE_ENCODER", {})
        feature_encoder.setdefault("LORA_RANK", None)
        feature_encoder.setdefault("LORA_ALPHA", None)

        iterative = waft.setdefault("ITERATIVE_MODULE", {})
        for key in ("PROP_ITER", "DELTA_ITER"):
            module_cfg = iterative.setdefault(key, {})
            module_cfg.setdefault("LORA_RANK", None)
            module_cfg.setdefault("LORA_ALPHA", None)

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        weights = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if weights is None:
            weights = checkpoint
        self.model.load_state_dict(weights, strict=False)

    def _sync_model_device(self, device: torch.device) -> None:
        model_device = next(self.model.parameters()).device
        if model_device != device:
            self.model.to(device)

    def _autocast_context(self, device: torch.device) -> Any:
        enabled = self.amp_dtype is not None and device.type == "cuda"
        if not enabled:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=self.amp_dtype, enabled=True)

    def _encoder_features(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        utils = importlib.import_module("model.utils")
        image1 = self.model.normalize_image(left.float())
        image2 = self.model.normalize_image(right.float())
        padder = utils.Padder(image1.shape, factor=self.model.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        fmap1, _fmap2, net = self.model.encoder(torch.stack([image1, image2], dim=1))
        return fmap1, net

    def _full_forward_with_features(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def capture_encoder(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> None:
            fmap1, _fmap2, net = output
            captured["features"] = (fmap1, net)

        handle = self.model.encoder.register_forward_hook(capture_encoder)
        try:
            output = self.model({"img1": left.float(), "img2": right.float()})
        finally:
            handle.remove()
        if "features" not in captured:
            raise RuntimeError("WAFT encoder hook did not capture features.")
        disp = output["disp_pred"].unsqueeze(1)
        fmap1, net = captured["features"]
        return fmap1, net, disp

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        self._sync_model_device(left.device)
        if self.freeze:
            self.model.eval()

        grad_ctx = torch.no_grad() if self.freeze else nullcontext()
        with grad_ctx, self._autocast_context(left.device):
            if self.use_disparity:
                fmap1, net, disp = self._full_forward_with_features(left, right)
            else:
                fmap1, net = self._encoder_features(left, right)
                disp = None

        all_features = {
            "fmap1": fmap1.to(dtype=left.dtype),
            "net": net.to(dtype=left.dtype),
        }
        if self.use_disparity:
            if disp is None:
                raise RuntimeError("WAFT disparity was requested but not produced.")
            all_features["disp"] = disp.float()
        out = {name: all_features[name] for name in self.feature_names}
        if self.use_disparity and "disp" not in out:
            out["disp"] = all_features["disp"]
        return out


__all__ = ["WAFTStereoBackbone"]
