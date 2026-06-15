from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .dino import DinoDenseFeatureBranch


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

    all_feature_names = ("fmap1", "net", "refine_net", "delta_block12", "disp", "dino")
    default_feature_names = ("fmap1", "net")
    uses_view_indices = True

    def __init__(
        self,
        waft_root: str | Path,
        config_path: str | Path,
        checkpoint_path: str | Path,
        freeze: bool = True,
        use_disparity: bool = True,
        feature_names: Sequence[str] | None = None,
        amp_dtype: str | None = None,
        dino: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.waft_root = Path(waft_root)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.freeze = bool(freeze)
        self.feature_names = self._validate_feature_names(feature_names)
        self.use_disparity = "disp" in self.feature_names
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)

        self._add_waft_root()
        self._patch_import_compat()
        waft_module = importlib.import_module("algorithms.waft")
        cfg = self._load_config()
        self.model = waft_module.WAFT(cfg)
        self._load_checkpoint()
        delta_feature_names = {"refine_net", "delta_block12"}
        if set(self.feature_names) & delta_feature_names and int(getattr(self.model, "iters", 0)) <= 0:
            requested = ", ".join(name for name in self.feature_names if name in delta_feature_names)
            raise ValueError(f"Selecting WAFT feature(s) {requested} requires at least one delta refine iteration.")

        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

        enc_dim = int(self.model.enc_dim)
        delta_dim = int(getattr(self.model.delta_decoder, "dim", enc_dim))
        self.max_disp = int(getattr(self.model, "max_disp", 0))
        self.feature_channels = {
            "fmap1": enc_dim,
            "net": enc_dim,
            "refine_net": enc_dim,
            "delta_block12": delta_dim,
            "disp": 1,
            "dino": enc_dim,
        }
        dino_cfg = dict(dino or {})
        self.dino_view_indices = tuple(int(idx) for idx in dino_cfg.get("view_indices", [0]))
        self.dino: DinoDenseFeatureBranch | None = None
        if "dino" in self.feature_names and self.dino_view_indices:
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
                output_channels=enc_dim,
                local_files_only=bool(dino_cfg.get("local_files_only", True)),
                freeze=bool(dino_cfg.get("freeze", True)),
                projection="linear",
            )
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
            timm_layers = importlib.import_module("timm.layers")
        except ModuleNotFoundError:
            timm_layers = importlib.import_module("timm.models.layers")
            sys.modules["timm.layers"] = timm_layers

        mlp = getattr(timm_layers, "Mlp", None)
        if mlp is not None and "use_conv" not in inspect.signature(mlp.__init__).parameters:

            class CompatMlp(nn.Module):
                def __init__(
                    self,
                    in_features: int,
                    hidden_features: int | None = None,
                    out_features: int | None = None,
                    act_layer: type[nn.Module] = nn.GELU,
                    bias: bool | tuple[bool, bool] = True,
                    drop: float | tuple[float, float] = 0.0,
                    use_conv: bool = False,
                ) -> None:
                    super().__init__()
                    out_features = out_features or in_features
                    hidden_features = hidden_features or in_features
                    if isinstance(bias, tuple):
                        bias1, bias2 = bias
                    else:
                        bias1 = bias2 = bias
                    if isinstance(drop, tuple):
                        drop1, drop2 = drop
                    else:
                        drop1 = drop2 = drop
                    linear_layer = nn.Conv2d if use_conv else nn.Linear
                    if use_conv:
                        self.fc1 = linear_layer(in_features, hidden_features, kernel_size=1, bias=bias1)
                        self.fc2 = linear_layer(hidden_features, out_features, kernel_size=1, bias=bias2)
                    else:
                        self.fc1 = linear_layer(in_features, hidden_features, bias=bias1)
                        self.fc2 = linear_layer(hidden_features, out_features, bias=bias2)
                    self.act = act_layer()
                    self.drop1 = nn.Dropout(drop1)
                    self.drop2 = nn.Dropout(drop2)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = self.fc1(x)
                    x = self.act(x)
                    x = self.drop1(x)
                    x = self.fc2(x)
                    x = self.drop2(x)
                    return x

            timm_layers.Mlp = CompatMlp

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

    def _dino_mask(self, view_indices: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.dino is None:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        if view_indices is None:
            return torch.ones(batch_size, device=device, dtype=torch.bool)
        if view_indices.shape[0] != batch_size:
            raise ValueError(f"view_indices must have length {batch_size}, got shape {tuple(view_indices.shape)}.")
        mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        view_indices = view_indices.to(device=device)
        for view_idx in self.dino_view_indices:
            mask = mask | (view_indices == view_idx)
        return mask

    def _dino_features(
        self,
        image: torch.Tensor,
        *,
        output_size: tuple[int, int] | None,
        view_indices: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = self._dino_mask(view_indices, image.shape[0], image.device)
        selected: torch.Tensor | None = None
        if mask.any():
            if self.dino is None:
                raise RuntimeError("DINO feature was requested but the DINO branch is not configured.")
            selected = self.dino(image[mask], output_size=output_size).to(dtype=dtype)
            feature_size = selected.shape[-2:]
        elif output_size is not None:
            feature_size = output_size
        else:
            patch_size = int(getattr(self.dino, "patch_size", 14))
            feature_size = (
                max(int(image.shape[-2]) // patch_size, 1),
                max(int(image.shape[-1]) // patch_size, 1),
            )
        features = torch.zeros(
            image.shape[0],
            self.feature_channels["dino"],
            feature_size[0],
            feature_size[1],
            device=image.device,
            dtype=dtype,
        )
        if selected is not None:
            features[mask] = selected
        return features

    def _encoder_features(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        utils = importlib.import_module("model.utils")
        image1 = self.model.normalize_image(left.float())
        image2 = self.model.normalize_image(right.float())
        padder = utils.Padder(image1.shape, factor=self.model.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        fmap1, _fmap2, net = self.model.encoder(torch.stack([image1, image2], dim=1))
        return fmap1, net

    def _reshape_delta_block12(self, tokens: torch.Tensor, input_shape: tuple[int, int]) -> torch.Tensor:
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected delta_block12 tokens shape [N,L,C], got {tuple(tokens.shape)}.")
        patch_size = int(getattr(self.model.delta_decoder, "patch_size", 0) or 0)
        if patch_size <= 0:
            raise RuntimeError("WAFT delta_decoder must expose a positive patch_size for delta_block12.")
        height = input_shape[0] // patch_size
        width = input_shape[1] // patch_size
        expected_tokens = height * width
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "WAFT delta_block12 token count does not match the expected spatial grid: "
                f"got {tokens.shape[1]} tokens, expected {expected_tokens} "
                f"from input_shape={input_shape} and patch_size={patch_size}."
            )
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)

    def _capture_delta_block12(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        captured: dict[str, torch.Tensor] = {}

        def capture_block(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"Expected delta_block12 hook output to be a tensor, got {type(output).__name__}.")
            captured["tokens"] = output

        block = self.model.delta_decoder.blks[-1]
        handle = block.register_forward_hook(capture_block)
        try:
            out = self.model.delta_decoder(inp)
        finally:
            handle.remove()
        tokens = captured.get("tokens")
        if tokens is None:
            raise RuntimeError("WAFT delta_block12 hook did not capture transformer block output.")
        return out, self._reshape_delta_block12(tokens, inp.shape[-2:])

    def _full_forward_with_features(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        utils = importlib.import_module("model.utils")
        image1 = self.model.normalize_image(left.float())
        image2 = self.model.normalize_image(right.float())
        padder = utils.Padder(image1.shape, factor=self.model.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)

        fmap1, fmap2, net = self.model.encoder(torch.stack([image1, image2], dim=1))
        encoder_net = net

        idx_bins_2x = torch.linspace(
            0,
            self.model.max_disp / 2,
            self.model.n_bins,
            device=fmap1.device,
            dtype=fmap1.dtype,
        ).view(1, self.model.n_bins, 1, 1)

        prop_hidden = self.model.prop_proj(torch.cat([fmap1, fmap2], dim=1))
        prop_hidden = self.model.prop_decoder(prop_hidden)
        prob_mask = 0.25 * self.model.prop_mask_head(prop_hidden)
        prob_bins = self.model.prop_bins_head(prop_hidden)
        prob_up = self.model.convex_upsample(prob_bins, prob_mask)
        prob_bins = F.softmax(prob_bins, dim=1)
        disp = torch.sum(prob_bins * idx_bins_2x, dim=1, keepdim=True)

        refine_net = None
        delta_block12 = None
        delta_disp_preds = []
        iters = int(self.model.iters)
        capture_delta_block12 = "delta_block12" in self.feature_names
        for itr in range(iters):
            disp = disp.detach()
            warped_fmap2 = utils.disp_warp(fmap2, disp, padding_mode="zeros")
            net = self.model.delta_proj(torch.cat([fmap1, warped_fmap2, net, disp], dim=1))
            if capture_delta_block12 and itr == iters - 1:
                net, delta_block12 = self._capture_delta_block12(net)
            else:
                net = self.model.delta_decoder(net)
            refine_net = net
            delta_disp = self.model.delta_disp_head(net)
            mask = 0.25 * self.model.delta_mask_head(net)
            disp = disp + delta_disp
            disp_up = self.model.convex_upsample(disp * 2, mask)
            delta_disp_preds.append(disp_up)

        if delta_disp_preds:
            disp_final = padder.unpad(delta_disp_preds[-1]).squeeze(1)
        else:
            idx_bins_1x = torch.linspace(
                0,
                self.model.max_disp,
                self.model.n_bins,
                device=fmap1.device,
                dtype=fmap1.dtype,
            ).view(1, self.model.n_bins, 1, 1)
            init = padder.unpad(prob_up)
            disp_final = torch.sum(F.softmax(init, dim=1) * idx_bins_1x, dim=1)
        return fmap1, encoder_net, refine_net, delta_block12, disp_final.unsqueeze(1)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        view_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._sync_model_device(left.device)
        if self.freeze:
            self.model.eval()

        grad_ctx = torch.no_grad() if self.freeze else nullcontext()
        with grad_ctx, self._autocast_context(left.device):
            if self.use_disparity or "refine_net" in self.feature_names or "delta_block12" in self.feature_names:
                fmap1, net, refine_net, delta_block12, disp = self._full_forward_with_features(left, right)
            else:
                fmap1, net = self._encoder_features(left, right)
                refine_net = None
                delta_block12 = None
                disp = None

        all_features = {
            "fmap1": fmap1.to(dtype=left.dtype),
            "net": net.to(dtype=left.dtype),
        }
        if refine_net is not None:
            all_features["refine_net"] = refine_net.to(dtype=left.dtype)
        elif "refine_net" in self.feature_names:
            raise RuntimeError("WAFT refine hidden feature was requested but not produced.")
        if delta_block12 is not None:
            all_features["delta_block12"] = delta_block12.to(dtype=left.dtype)
        elif "delta_block12" in self.feature_names:
            raise RuntimeError("WAFT delta block 12 feature was requested but not produced.")
        if self.use_disparity:
            if disp is None:
                raise RuntimeError("WAFT disparity was requested but not produced.")
            all_features["disp"] = disp.float()
        if "dino" in self.feature_names:
            all_features["dino"] = self._dino_features(
                left,
                output_size=None,
                view_indices=view_indices,
                dtype=all_features["fmap1"].dtype,
            )
        out = {name: all_features[name] for name in self.feature_names}
        return out


__all__ = ["WAFTStereoBackbone"]
