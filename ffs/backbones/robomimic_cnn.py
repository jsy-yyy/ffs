from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import models as vision_models

from .dino import DinoDenseFeatureBranch


OBS_KEY_MODALITIES = {
    "robot0_eef_pos": "low_dim",
    "robot0_eef_quat": "low_dim",
    "robot0_gripper_qpos": "low_dim",
    "object": "low_dim",
    "agentview_image": "rgb",
    "agentview_right_image": "rgb",
    "agentview_disp": "rgb",
    "agentview_dino": "rgb",
    "robot0_eye_in_hand_image": "rgb",
    "robot0_eye_in_hand_right_image": "rgb",
    "robot0_eye_in_hand_disp": "rgb",
    "robot0_eye_in_hand_dino": "rgb",
}


def _center_crop_chw(inputs: torch.Tensor, crop_height: int, crop_width: int) -> torch.Tensor:
    height, width = inputs.shape[-2:]
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return inputs[..., top:top + crop_height, left:left + crop_width]


def _random_crop_chw(
    inputs: torch.Tensor,
    crop_height: int,
    crop_width: int,
    num_crops: int,
    pos_enc: bool,
) -> torch.Tensor:
    batch = inputs.shape[0]
    height, width = inputs.shape[-2:]
    max_top = height - crop_height
    max_left = width - crop_width
    top = torch.randint(0, max_top + 1, (batch, num_crops), device=inputs.device)
    left = torch.randint(0, max_left + 1, (batch, num_crops), device=inputs.device)
    crops = []
    for batch_idx in range(batch):
        for crop_idx in range(num_crops):
            t = int(top[batch_idx, crop_idx].item())
            l = int(left[batch_idx, crop_idx].item())
            crop = inputs[batch_idx, :, t:t + crop_height, l:l + crop_width]
            if pos_enc:
                yy = torch.linspace(
                    t / max(height - 1, 1),
                    (t + crop_height - 1) / max(height - 1, 1),
                    crop_height,
                    device=inputs.device,
                    dtype=inputs.dtype,
                )
                xx = torch.linspace(
                    l / max(width - 1, 1),
                    (l + crop_width - 1) / max(width - 1, 1),
                    crop_width,
                    device=inputs.device,
                    dtype=inputs.dtype,
                )
                grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
                crop = torch.cat([crop, grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0)
            crops.append(crop)
    return torch.stack(crops, dim=0)


class ResNet18Conv(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        pretrained: bool = False,
        input_coord_conv: bool = False,
    ) -> None:
        super().__init__()
        if input_coord_conv:
            raise NotImplementedError("ResNet18Conv input_coord_conv is not supported in the local FFS copy.")
        weights = vision_models.ResNet18_Weights.DEFAULT if pretrained else None
        net = vision_models.resnet18(weights=weights)
        if input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self._input_channel = input_channel
        self._input_coord_conv = input_coord_conv
        self.nets = nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape: tuple[int, int, int]) -> list[int]:
        if len(input_shape) != 3:
            raise ValueError(f"ResNet18Conv expected CHW input shape, got {input_shape}.")
        return [512, int(np.ceil(input_shape[1] / 32.0)), int(np.ceil(input_shape[2] / 32.0))]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.nets(inputs)


class CropRandomizer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        crop_height: int = 216,
        crop_width: int = 216,
        num_crops: int = 1,
        pos_enc: bool = False,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"CropRandomizer expected CHW input shape, got {input_shape}.")
        if crop_height >= input_shape[1] or crop_width >= input_shape[2]:
            raise ValueError(f"Crop size {(crop_height, crop_width)} must be smaller than input shape {input_shape}.")
        self.input_shape = tuple(input_shape)
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width)
        self.num_crops = int(num_crops)
        self.pos_enc = bool(pos_enc)

    def output_shape_in(self, input_shape: tuple[int, int, int] | None = None) -> list[int]:
        input_shape = tuple(input_shape or self.input_shape)
        out_c = input_shape[0] + 2 if self.pos_enc else input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape: tuple[int, ...] | list[int] | None = None) -> list[int]:
        return list(input_shape or self.output_shape_in(self.input_shape))

    def forward_in(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            return _random_crop_chw(inputs, self.crop_height, self.crop_width, self.num_crops, self.pos_enc)
        return _center_crop_chw(inputs, self.crop_height, self.crop_width)

    def forward_out(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return inputs
        batch_size = inputs.shape[0] // self.num_crops
        return inputs.reshape(batch_size, self.num_crops, *inputs.shape[1:]).mean(dim=1)


def _replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16) -> nn.Module:
    for name, child in list(root_module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = max(1, child.num_features // features_per_group)
            setattr(root_module, name, nn.GroupNorm(num_groups=groups, num_channels=child.num_features))
        else:
            _replace_bn_with_gn(child, features_per_group=features_per_group)
    return root_module


class ObservationFeatureMapEncoder(nn.Module):
    def __init__(
        self,
        obs_shapes: OrderedDict[str, tuple[int, ...]],
        rgb_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        rgb_cfg = dict(rgb_cfg or {})
        self.obs_shapes = obs_shapes
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_channels: dict[str, int] = {}

        for key, shape in self.obs_shapes.items():
            if OBS_KEY_MODALITIES[key] != "rgb":
                raise ValueError(f"ObservationFeatureMapEncoder only supports rgb-like keys, got {key!r}.")
            if rgb_cfg.get("backbone_class", "ResNet18Conv") != "ResNet18Conv":
                raise ValueError("Local CNN backbone only supports backbone_class='ResNet18Conv'.")
            randomizer = CropRandomizer(
                input_shape=tuple(shape),
                crop_height=int(rgb_cfg.get("crop_height", 216)),
                crop_width=int(rgb_cfg.get("crop_width", 216)),
                num_crops=int(rgb_cfg.get("num_crops", 1)),
                pos_enc=bool(rgb_cfg.get("pos_enc", False)),
            )
            randomized_shape = randomizer.output_shape_in(tuple(shape))
            self.obs_randomizers[key] = nn.ModuleList([randomizer])
            net = ResNet18Conv(
                input_channel=int(randomized_shape[0]),
                pretrained=bool(rgb_cfg.get("pretrained", False)),
                input_coord_conv=bool(rgb_cfg.get("input_coord_conv", False)),
            )
            self.obs_nets[key] = net
            self.feature_channels[key] = int(net.output_shape(tuple(randomized_shape))[0])

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not set(self.obs_shapes).issubset(obs_dict):
            raise ValueError(f"ObservationFeatureMapEncoder missing keys: {sorted(set(self.obs_shapes) - set(obs_dict))}.")
        features = {}
        for key in self.obs_shapes:
            x = obs_dict[key]
            for rand in self.obs_randomizers[key]:
                x = rand.forward_in(x)
            x = self.obs_nets[key](x)
            for rand in reversed(self.obs_randomizers[key]):
                x = rand.forward_out(x)
            features[key] = x
        return features


class RobomimicCNNBackbone(nn.Module):
    dino_feature_names_by_view = {
        0: "agentview_dino",
        1: "robot0_eye_in_hand_dino",
    }

    def __init__(
        self,
        state_dim: int,
        observation_horizon: int = 2,
        image_size: list[int] | tuple[int, int] = (224, 224),
        use_left_only: bool = True,
        use_disparity: bool = False,
        rgb_cfg: dict[str, Any] | None = None,
        rgb_encoder_cfg: dict[str, Any] | None = None,
        dino: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if state_dim != 23:
            raise ValueError(f"RobomimicCNNBackbone expects 23D state, got state_dim={state_dim}.")
        self.state_dim = state_dim
        self.observation_horizon = observation_horizon
        self.image_size = tuple(image_size)
        self.use_left_only = bool(use_left_only)
        self.use_disparity = bool(use_disparity)
        self.expects_sequence_input = True

        height, width = self.image_size
        image_shapes = OrderedDict(
            [
                ("agentview_image", (3, height, width)),
                *([("agentview_disp", (1, height, width))] if self.use_disparity else []),
                ("robot0_eye_in_hand_image", (3, height, width)),
                *([("robot0_eye_in_hand_disp", (1, height, width))] if self.use_disparity else []),
            ]
        )
        if not self.use_left_only:
            image_shapes["agentview_right_image"] = (3, height, width)
            image_shapes["robot0_eye_in_hand_right_image"] = (3, height, width)
        self.image_shapes = image_shapes
        encoder_cfg = dict(rgb_encoder_cfg or rgb_cfg or {})
        self.obs_encoder = ObservationFeatureMapEncoder(self.image_shapes, rgb_cfg=encoder_cfg)
        if bool(encoder_cfg.get("replace_bn_with_gn", True)):
            self.obs_encoder = _replace_bn_with_gn(self.obs_encoder)
        dino_cfg = dict(dino or {})
        self.dino_enabled = bool(dino_cfg.get("enabled", False))
        self.dino_view_indices = tuple(int(idx) for idx in dino_cfg.get("view_indices", [0, 1]))
        self.dino_feature_channels = int(dino_cfg.get("output_channels", 128))
        self.dino: DinoDenseFeatureBranch | None = None
        self.dino_feature_names = (
            tuple(
                self.dino_feature_names_by_view[idx]
                for idx in self.dino_view_indices
                if idx in self.dino_feature_names_by_view
            )
            if self.dino_enabled
            else ()
        )
        if self.dino_enabled and len(self.dino_feature_names) != len(self.dino_view_indices):
            supported = ", ".join(str(idx) for idx in sorted(self.dino_feature_names_by_view))
            raise ValueError(f"backbone.dino.view_indices only supports native DP views: {supported}.")
        if self.dino_enabled and self.dino_view_indices:
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
                output_channels=self.dino_feature_channels,
                local_files_only=bool(dino_cfg.get("local_files_only", True)),
                freeze=bool(dino_cfg.get("freeze", True)),
                projection="linear",
            )
        self.feature_names = (*self.image_shapes.keys(), *self.dino_feature_names)
        self.feature_channels = dict(self.obs_encoder.feature_channels)
        self.feature_channels.update({name: self.dino_feature_channels for name in self.dino_feature_names})
        self.feature_view_counts = {name: 1 for name in self.feature_names}
        self.output_dim = None

    @staticmethod
    def split_state(state: torch.Tensor) -> dict[str, torch.Tensor]:
        if state.shape[-1] != 23:
            raise ValueError(f"Expected state last dim 23, got {state.shape[-1]}.")
        return {
            "robot0_eef_pos": state[..., 0:3],
            "robot0_eef_quat": state[..., 3:7],
            "robot0_gripper_qpos": state[..., 7:9],
            "object": state[..., 9:23],
        }

    def make_obs_dict(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        disparity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if left.shape[2] < 2:
            raise ValueError("RobomimicCNNBackbone expects two stereo camera pairs.")
        if not self.use_left_only and right.shape[2] < 2:
            raise ValueError("RobomimicCNNBackbone expects two right-camera views when use_left_only=false.")
        if self.use_disparity:
            if disparity is None:
                raise ValueError("RobomimicCNNBackbone requires disparity when use_disparity=true.")
            if disparity.ndim != 6 or disparity.shape[2] < 2 or disparity.shape[3] != 1:
                raise ValueError(
                    "RobomimicCNNBackbone expects disparity shape [B,T,V,1,H,W] "
                    f"with at least two views, got {tuple(disparity.shape)}."
                )
            if disparity.shape[:2] != left.shape[:2]:
                raise ValueError(
                    "Disparity batch/time dimensions must match left images: "
                    f"disparity={tuple(disparity.shape[:2])} left={tuple(left.shape[:2])}."
                )
        obs = self.split_state(state)
        obs["agentview_image"] = (left[:, :, 0].float() / 255.0).clamp(0.0, 1.0)
        obs["robot0_eye_in_hand_image"] = (left[:, :, 1].float() / 255.0).clamp(0.0, 1.0)
        if self.use_disparity:
            obs["agentview_disp"] = disparity[:, :, 0].float().clamp(0.0, 1.0)
            obs["robot0_eye_in_hand_disp"] = disparity[:, :, 1].float().clamp(0.0, 1.0)
        if not self.use_left_only:
            obs["agentview_right_image"] = (right[:, :, 0].float() / 255.0).clamp(0.0, 1.0)
            obs["robot0_eye_in_hand_right_image"] = (right[:, :, 1].float() / 255.0).clamp(0.0, 1.0)
        return obs

    def _dino_features(self, left: torch.Tensor, batch: int, time: int) -> dict[str, torch.Tensor]:
        if not self.dino_enabled or not self.dino_feature_names:
            return {}
        if self.dino is None:
            raise RuntimeError("DINO features are enabled but the DINO branch is not configured.")

        out: dict[str, torch.Tensor] = {}
        patch_size = int(getattr(self.dino, "patch_size", 14))
        output_size = (
            max(int(left.shape[-2]) // patch_size, 1),
            max(int(left.shape[-1]) // patch_size, 1),
        )
        for view_idx, name in zip(self.dino_view_indices, self.dino_feature_names):
            image = left[:, :, view_idx].reshape(batch * time, *left.shape[3:])
            feature = self.dino(image, output_size=output_size)
            out[name] = feature.view(batch, time, *feature.shape[1:])
        return out

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        state: torch.Tensor,
        disparity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        obs = self.make_obs_dict(left, right, state, disparity)
        batch, time = state.shape[:2]
        if time != self.observation_horizon:
            raise ValueError(
                f"RobomimicCNNBackbone expected observation_horizon={self.observation_horizon}, got {time}."
            )
        flat_obs = {
            key: value.reshape(batch * time, *value.shape[2:])
            for key, value in obs.items()
            if key in self.image_shapes
        }
        flat_features = self.obs_encoder(flat_obs)
        features = {
            key: value.view(batch, time, *value.shape[1:])
            for key, value in flat_features.items()
        }
        features.update(self._dino_features(left, batch, time))
        return features


__all__ = [
    "CropRandomizer",
    "ObservationFeatureMapEncoder",
    "ResNet18Conv",
    "RobomimicCNNBackbone",
]
