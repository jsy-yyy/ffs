from __future__ import annotations

from typing import Any

import torch.nn as nn

from ffs.adapters import build_adapter
from ffs.backbones import FoundationStereoBackbone, RobomimicCNNBackbone, StereoTransformerBackbone, WAFTStereoBackbone
from ffs.heads import DiffusionUNetActionHead, build_action_head
from ffs.policies.backbone_adapter_head import BackboneAdapterHeadPolicy


def resolve_action_head_cfg(head_cfg: dict[str, Any]) -> dict[str, Any]:
    head_type = head_cfg.get("type", "mlp")
    nested_cfg = head_cfg.get(head_type)
    if isinstance(nested_cfg, dict):
        return {"type": head_type, **nested_cfg}
    return {"type": head_type}


def _feature_names_with_disp(feature_names: object, default: list[str]) -> list[str]:
    if feature_names is None:
        names = list(default)
    elif isinstance(feature_names, str):
        names = [feature_names]
    else:
        names = list(feature_names)  # type: ignore[arg-type]
    if "disp" not in names:
        names.append("disp")
    return names


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
            feature_names=_feature_names_with_disp(ffs_cfg.get("feature_names"), ["feat_04"]),
        )
        return provider, max_disp

    waft_cfg = dict(disparity_cfg.get("waft") or {})
    provider = WAFTStereoBackbone(
        waft_root=waft_cfg["waft_root"],
        config_path=waft_cfg["config_path"],
        checkpoint_path=waft_cfg["checkpoint_path"],
        freeze=freeze,
        use_disparity=True,
        feature_names=_feature_names_with_disp(waft_cfg.get("feature_names"), ["fmap1"]),
        amp_dtype=waft_cfg.get("amp_dtype"),
    )
    return provider, max_disp


def _build_backbone(
    *,
    backbone_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    adapter_cfg: dict[str, Any],
    image_size: list[int] | tuple[int, int],
) -> tuple[nn.Module, nn.Module | None, int | None]:
    backbone_type = backbone_cfg.get("type", "ffs-based")
    disparity_provider: nn.Module | None = None
    disparity_max_disp: int | None = None

    if backbone_type == "cnn-based":
        disparity_provider, disparity_max_disp = build_cnn_disparity_provider(backbone_cfg)
        use_disparity = bool(backbone_cfg.get("use_disparity", disparity_provider is not None))
        if use_disparity and disparity_provider is None:
            raise ValueError("backbone.type='cnn-based' with use_disparity=true requires backbone.disparity.enabled=true.")
        observation_horizon = int(policy_cfg.get("observation_horizon", policy_cfg["num_history_frames"]))
        backbone = RobomimicCNNBackbone(
            state_dim=int(policy_cfg["state_dim"]),
            observation_horizon=observation_horizon,
            image_size=backbone_cfg.get("image_size", image_size),
            use_left_only=bool(backbone_cfg.get("use_left_only", True)),
            use_disparity=use_disparity,
            rgb_encoder_cfg=backbone_cfg.get("rgb_encoder"),
            dino=backbone_cfg.get("dino"),
        )
        return backbone, disparity_provider, disparity_max_disp

    if backbone_type not in {"ffs-based", "waft-based", "stereo-transformer"}:
        raise ValueError("backbone.type must be one of: cnn-based, ffs-based, waft-based, stereo-transformer.")

    backbone_feature_names = tuple(backbone_cfg.get("feature_names", ()))
    backbone_use_disparity = "disp" in backbone_feature_names
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
    elif backbone_type == "waft-based":
        backbone = WAFTStereoBackbone(
            waft_root=backbone_cfg["waft_root"],
            config_path=backbone_cfg["config_path"],
            checkpoint_path=backbone_cfg["checkpoint_path"],
            freeze=backbone_cfg.get("freeze", True),
            use_disparity=backbone_use_disparity,
            feature_names=backbone_cfg["feature_names"],
            amp_dtype=backbone_cfg.get("amp_dtype"),
            dino=backbone_cfg.get("dino"),
        )
    else:
        if backbone_use_disparity:
            raise ValueError("backbone.type='stereo-transformer' does not produce disparity.")
        backbone = StereoTransformerBackbone(
            feature_names=backbone_cfg.get("feature_names", ["stereo_latent"]),
            cnn_feature_dim=int(backbone_cfg.get("cnn_feature_dim", 128)),
            token_dim=int(backbone_cfg.get("token_dim", 256)),
            latent_dim=int(backbone_cfg.get("latent_dim", 128)),
            num_layers=int(backbone_cfg.get("num_layers", 2)),
            num_heads=int(backbone_cfg.get("num_heads", 8)),
            mlp_ratio=float(backbone_cfg.get("mlp_ratio", 4.0)),
            dropout=float(backbone_cfg.get("dropout", 0.0)),
            pretrained_resnet=bool(backbone_cfg.get("pretrained_resnet", False)),
            freeze_cnn=bool(backbone_cfg.get("freeze_cnn", False)),
            dino=backbone_cfg.get("dino"),
        )
    return backbone, disparity_provider, disparity_max_disp


def _build_diffusion_head(
    *,
    head_cfg: dict[str, Any],
    cond_dim: int,
    policy_cfg: dict[str, Any],
) -> DiffusionUNetActionHead:
    head_nested = dict(head_cfg.get("diffusion_unet") or {})
    ddpm_cfg = dict(head_nested.pop("ddpm", {}) or {})
    observation_horizon = int(policy_cfg.get("observation_horizon", policy_cfg["num_history_frames"]))
    return DiffusionUNetActionHead(
        cond_dim=cond_dim,
        action_dim=int(policy_cfg["action_dim"]),
        observation_horizon=observation_horizon,
        action_horizon=int(policy_cfg["action_horizon"]),
        prediction_horizon=int(policy_cfg.get("prediction_horizon", policy_cfg["action_horizon"])),
        diffusion_step_embed_dim=int(head_nested.get("diffusion_step_embed_dim", 256)),
        down_dims=head_nested.get("down_dims", [256, 512, 1024]),
        kernel_size=int(head_nested.get("kernel_size", 5)),
        n_groups=int(head_nested.get("n_groups", 8)),
        ddpm_cfg=ddpm_cfg,
    )


def _build_head(
    *,
    head_cfg: dict[str, Any],
    adapter: nn.Module,
    policy_cfg: dict[str, Any],
) -> nn.Module:
    head_type = head_cfg.get("type", "mlp")
    if head_type == "diffusion_unet":
        if getattr(adapter, "output_kind", None) != "cond":
            raise ValueError("head.type='diffusion_unet' requires an adapter with output_kind='cond'.")
        return _build_diffusion_head(head_cfg=head_cfg, cond_dim=int(adapter.cond_dim), policy_cfg=policy_cfg)

    if getattr(adapter, "output_kind", None) != "tokens":
        raise ValueError(f"head.type={head_type!r} requires an adapter with output_kind='tokens'.")
    action_head_cfg = resolve_action_head_cfg(head_cfg)
    return build_action_head(
        action_head_cfg,
        input_dim=int(adapter.condition_len) * int(adapter.token_dim),
        action_dim=int(policy_cfg["action_dim"]),
        action_horizon=int(policy_cfg["action_horizon"]),
        prediction_horizon=int(policy_cfg.get("prediction_horizon", policy_cfg["action_horizon"])),
        frame_token_dim=int(adapter.token_dim),
        condition_len=int(adapter.condition_len),
        num_history_frames=int(policy_cfg["num_history_frames"]),
        tokens_per_frame=int(adapter.tokens_per_frame),
    )


def build_policy(cfg: dict[str, Any]) -> BackboneAdapterHeadPolicy:
    backbone_cfg = cfg["backbone"]
    policy_cfg = cfg["policy"]
    dataset_cfg = cfg.get("dataset", {})
    adapter_cfg = cfg.get("adapter")
    if not isinstance(adapter_cfg, dict):
        raise ValueError("adapter section is required in the new backbone/adapter/head schema.")
    head_cfg = cfg.get("head", {})

    backbone, disparity_provider, disparity_max_disp = _build_backbone(
        backbone_cfg=backbone_cfg,
        policy_cfg=policy_cfg,
        adapter_cfg=adapter_cfg,
        image_size=dataset_cfg.get("image_size", [224, 224]),
    )
    adapter = build_adapter(adapter_cfg, backbone=backbone, policy_cfg=policy_cfg, dataset_cfg=dataset_cfg)
    action_head = _build_head(head_cfg=head_cfg, adapter=adapter, policy_cfg=policy_cfg)
    return BackboneAdapterHeadPolicy(
        backbone=backbone,
        adapter=adapter,
        action_head=action_head,
        num_history_frames=int(policy_cfg.get("observation_horizon", policy_cfg["num_history_frames"])),
        state_dim=int(policy_cfg["state_dim"]),
        num_stereo_pairs=len(dataset_cfg.get("camera_pairs", [])),
        disparity_provider=disparity_provider,
        disparity_max_disp=disparity_max_disp,
        disparity_ablation=policy_cfg.get("disparity_ablation", "none"),
    )


__all__ = [
    "build_policy",
]
