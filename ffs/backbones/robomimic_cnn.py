from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_models


OBS_KEY_MODALITIES = {
    "robot0_eef_pos": "low_dim",
    "robot0_eef_quat": "low_dim",
    "robot0_gripper_qpos": "low_dim",
    "object": "low_dim",
    "agentview_image": "rgb",
    "robot0_eye_in_hand_image": "rgb",
}


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


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


class SpatialSoftmax(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_kp: int | None = 32,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        output_variance: bool = False,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"SpatialSoftmax expected CHW input shape, got {input_shape}.")
        self._in_c, self._in_h, self._in_w = input_shape
        self._num_kp = int(num_kp) if num_kp is not None else self._in_c
        self.nets = nn.Conv2d(self._in_c, self._num_kp, kernel_size=1) if num_kp is not None else None
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std
        temp = torch.ones(1) * float(temperature)
        if learnable_temperature:
            self.register_parameter("temperature", nn.Parameter(temp, requires_grad=True))
        else:
            self.register_buffer("temperature", temp)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        self.register_buffer("pos_x", torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float())
        self.register_buffer("pos_y", torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float())
        self.kps: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None

    def output_shape(self, input_shape: tuple[int, int, int]) -> list[int]:
        if len(input_shape) != 3 or input_shape[0] != self._in_c:
            raise ValueError(f"SpatialSoftmax got incompatible input shape {input_shape}.")
        return [self._num_kp, 2]

    def forward(self, feature: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if feature.shape[1:] != (self._in_c, self._in_h, self._in_w):
            raise ValueError(
                f"SpatialSoftmax expected [B,{self._in_c},{self._in_h},{self._in_w}], got {tuple(feature.shape)}."
            )
        if self.nets is not None:
            feature = self.nets(feature)

        attention = F.softmax(feature.reshape(-1, self._in_h * self._in_w) / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        feature_keypoints = torch.cat([expected_x, expected_y], dim=1).view(-1, self._num_kp, 2)
        if self.training and self.noise_std > 0:
            feature_keypoints = feature_keypoints + torch.randn_like(feature_keypoints) * self.noise_std

        if self.output_variance:
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], dim=1).reshape(-1, self._num_kp, 2, 2)
            self.kps = (feature_keypoints.detach(), feature_covar.detach())
            return feature_keypoints, feature_covar

        self.kps = feature_keypoints.detach()
        return feature_keypoints


class VisualCore(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        backbone_class: str = "ResNet18Conv",
        pool_class: str = "SpatialSoftmax",
        backbone_kwargs: dict[str, Any] | None = None,
        pool_kwargs: dict[str, Any] | None = None,
        flatten: bool = True,
        feature_dimension: int | None = 64,
    ) -> None:
        super().__init__()
        if backbone_class != "ResNet18Conv":
            raise ValueError("Local VisualCore only supports backbone_class='ResNet18Conv'.")
        if pool_class not in {"SpatialSoftmax", None}:
            raise ValueError("Local VisualCore only supports pool_class='SpatialSoftmax' or None.")
        self.input_shape = tuple(input_shape)
        self.flatten = bool(flatten)
        backbone_kwargs = dict(backbone_kwargs or {})
        backbone_kwargs["input_channel"] = self.input_shape[0]
        self.backbone = ResNet18Conv(**backbone_kwargs)
        feat_shape = self.backbone.output_shape(self.input_shape)
        net_list: list[nn.Module] = [self.backbone]

        self.pool: SpatialSoftmax | None = None
        if pool_class is not None:
            pool_kwargs = dict(pool_kwargs or {})
            pool_kwargs["input_shape"] = tuple(feat_shape)
            self.pool = SpatialSoftmax(**pool_kwargs)
            feat_shape = self.pool.output_shape(tuple(feat_shape))
            net_list.append(self.pool)
        if self.flatten:
            net_list.append(nn.Flatten(start_dim=1, end_dim=-1))
        self.feature_dimension = feature_dimension
        if feature_dimension is not None:
            if not self.flatten:
                raise ValueError("feature_dimension requires flatten=True.")
            net_list.append(nn.Linear(int(np.prod(feat_shape)), int(feature_dimension)))
        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape: tuple[int, int, int]) -> list[int]:
        if self.feature_dimension is not None:
            return [int(self.feature_dimension)]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            feat_shape = self.pool.output_shape(tuple(feat_shape))
        if self.flatten:
            return [int(np.prod(feat_shape))]
        return list(feat_shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape[-3:]) != self.input_shape:
            raise ValueError(f"VisualCore expected input shape {self.input_shape}, got {tuple(inputs.shape[-3:])}.")
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


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        obs_shapes: OrderedDict[str, tuple[int, ...]],
        rgb_cfg: dict[str, Any] | None = None,
        feature_activation: type[nn.Module] | None = nn.ReLU,
    ) -> None:
        super().__init__()
        rgb_cfg = dict(rgb_cfg or {})
        self.obs_shapes = obs_shapes
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_activation = feature_activation() if feature_activation is not None else None

        for key, shape in self.obs_shapes.items():
            modality = OBS_KEY_MODALITIES[key]
            if modality == "rgb":
                randomizer = CropRandomizer(
                    input_shape=tuple(shape),
                    crop_height=int(rgb_cfg.get("crop_height", 216)),
                    crop_width=int(rgb_cfg.get("crop_width", 216)),
                    num_crops=int(rgb_cfg.get("num_crops", 1)),
                    pos_enc=bool(rgb_cfg.get("pos_enc", False)),
                )
                randomized_shape = randomizer.output_shape_in(tuple(shape))
                self.obs_randomizers[key] = nn.ModuleList([randomizer])
                self.obs_nets[key] = VisualCore(
                    input_shape=tuple(randomized_shape),
                    backbone_class=rgb_cfg.get("backbone_class", "ResNet18Conv"),
                    pool_class=rgb_cfg.get("pool_class", "SpatialSoftmax"),
                    backbone_kwargs={
                        "pretrained": bool(rgb_cfg.get("pretrained", False)),
                        "input_coord_conv": bool(rgb_cfg.get("input_coord_conv", False)),
                    },
                    pool_kwargs={
                        "num_kp": int(rgb_cfg.get("num_kp", 32)),
                        "learnable_temperature": bool(rgb_cfg.get("learnable_temperature", False)),
                        "temperature": float(rgb_cfg.get("temperature", 1.0)),
                        "noise_std": float(rgb_cfg.get("noise_std", 0.0)),
                    },
                    feature_dimension=int(rgb_cfg.get("feature_dimension", 64)),
                )
            else:
                self.obs_randomizers[key] = nn.ModuleList([])
                self.obs_nets[key] = nn.Identity()

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if not set(self.obs_shapes).issubset(obs_dict):
            raise ValueError(f"ObservationEncoder missing keys: {sorted(set(self.obs_shapes) - set(obs_dict))}.")
        feats = []
        for key, shape in self.obs_shapes.items():
            x = obs_dict[key]
            for rand in self.obs_randomizers[key]:
                x = rand.forward_in(x)
            if OBS_KEY_MODALITIES[key] == "rgb":
                x = self.obs_nets[key](x)
                if self.feature_activation is not None:
                    x = self.feature_activation(x)
            for rand in reversed(self.obs_randomizers[key]):
                x = rand.forward_out(x)
            feats.append(_flatten_batch(x))
        return torch.cat(feats, dim=-1)

    def output_shape(self) -> list[int]:
        feat_dim = 0
        for key, shape in self.obs_shapes.items():
            feat_shape: list[int] | tuple[int, ...] = shape
            for rand in self.obs_randomizers[key]:
                feat_shape = rand.output_shape_in(tuple(feat_shape))
            if OBS_KEY_MODALITIES[key] == "rgb":
                feat_shape = self.obs_nets[key].output_shape(tuple(feat_shape))
            for rand in self.obs_randomizers[key]:
                feat_shape = rand.output_shape_out(feat_shape)
            feat_dim += int(np.prod(feat_shape))
        return [feat_dim]


class ObservationGroupEncoder(nn.Module):
    def __init__(
        self,
        observation_group_shapes: OrderedDict[str, OrderedDict[str, tuple[int, ...]]],
        rgb_cfg: dict[str, Any] | None = None,
        feature_activation: type[nn.Module] | None = nn.ReLU,
    ) -> None:
        super().__init__()
        self.observation_group_shapes = observation_group_shapes
        self.nets = nn.ModuleDict(
            {
                obs_group: ObservationEncoder(obs_shapes, rgb_cfg=rgb_cfg, feature_activation=feature_activation)
                for obs_group, obs_shapes in self.observation_group_shapes.items()
            }
        )

    def forward(self, **inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not set(self.observation_group_shapes).issubset(inputs):
            raise ValueError(
                f"ObservationGroupEncoder missing groups: {sorted(set(self.observation_group_shapes) - set(inputs))}."
            )
        return torch.cat([self.nets[obs_group](inputs[obs_group]) for obs_group in self.observation_group_shapes], dim=-1)

    def output_shape(self) -> list[int]:
        return [sum(self.nets[obs_group].output_shape()[0] for obs_group in self.observation_group_shapes)]


def _replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16) -> nn.Module:
    for name, child in list(root_module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = max(1, child.num_features // features_per_group)
            setattr(root_module, name, nn.GroupNorm(num_groups=groups, num_channels=child.num_features))
        else:
            _replace_bn_with_gn(child, features_per_group=features_per_group)
    return root_module


class RobomimicCNNBackbone(nn.Module):
    def __init__(
        self,
        state_dim: int,
        observation_horizon: int = 2,
        image_size: list[int] | tuple[int, int] = (224, 224),
        use_left_only: bool = True,
        rgb_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if state_dim != 23:
            raise ValueError(f"RobomimicCNNBackbone expects 23D state, got state_dim={state_dim}.")
        self.state_dim = state_dim
        self.observation_horizon = observation_horizon
        self.image_size = tuple(image_size)
        self.use_left_only = bool(use_left_only)
        if not self.use_left_only:
            raise NotImplementedError("RobomimicCNNBackbone currently supports use_left_only=true only.")

        height, width = self.image_size
        observation_group_shapes = OrderedDict(
            [
                (
                    "obs",
                    OrderedDict(
                        [
                            ("agentview_image", (3, height, width)),
                            ("object", (14,)),
                            ("robot0_eef_pos", (3,)),
                            ("robot0_eef_quat", (4,)),
                            ("robot0_eye_in_hand_image", (3, height, width)),
                            ("robot0_gripper_qpos", (2,)),
                        ]
                    ),
                )
            ]
        )
        self.obs_encoder = ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            rgb_cfg=rgb_cfg,
        )
        rgb_cfg = dict(rgb_cfg or {})
        if bool(rgb_cfg.get("replace_bn_with_gn", True)):
            self.obs_encoder = _replace_bn_with_gn(self.obs_encoder)
        self.output_dim = int(self.obs_encoder.output_shape()[0]) * observation_horizon

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

    def make_obs_dict(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        del right
        if left.shape[2] < 2:
            raise ValueError("RobomimicCNNBackbone expects two stereo camera pairs.")
        obs = self.split_state(state)
        obs["agentview_image"] = (left[:, :, 0].float() / 255.0).clamp(0.0, 1.0)
        obs["robot0_eye_in_hand_image"] = (left[:, :, 1].float() / 255.0).clamp(0.0, 1.0)
        return obs

    def forward(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        obs = self.make_obs_dict(left, right, state)
        batch, time = state.shape[:2]
        if time != self.observation_horizon:
            raise ValueError(
                f"RobomimicCNNBackbone expected observation_horizon={self.observation_horizon}, got {time}."
            )
        flat_obs = {
            key: value.reshape(batch * time, *value.shape[2:])
            for key, value in obs.items()
        }
        obs_features = self.obs_encoder(obs=flat_obs).view(batch, time, -1)
        return obs_features.flatten(start_dim=1)


__all__ = [
    "CropRandomizer",
    "ObservationEncoder",
    "ObservationGroupEncoder",
    "ResNet18Conv",
    "RobomimicCNNBackbone",
    "SpatialSoftmax",
    "VisualCore",
]
