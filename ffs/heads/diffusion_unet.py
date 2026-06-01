from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0]
        bias = embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: list[int] | tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = int(down_dims[0])
        dsed = diffusion_step_embed_dim
        cond_dim = dsed + global_cond_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ]
        )
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample = sample.moveaxis(-1, -2)
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timesteps = timestep[None].to(sample.device)
        else:
            timesteps = timestep.to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        return self.final_conv(x).moveaxis(-1, -2)


def _squaredcos_cap_v2_betas(num_train_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    def alpha_bar(time_step: float) -> float:
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


@dataclass
class DDPMSchedulerOutput:
    prev_sample: torch.Tensor


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
    ) -> None:
        if beta_schedule != "squaredcos_cap_v2":
            raise ValueError("Only beta_schedule='squaredcos_cap_v2' is supported.")
        if prediction_type != "epsilon":
            raise ValueError("Only prediction_type='epsilon' is supported.")
        self.num_train_timesteps = int(num_train_timesteps)
        self.clip_sample = bool(clip_sample)
        self.prediction_type = prediction_type
        self.betas = _squaredcos_cap_v2_betas(self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0, dtype=torch.float32)
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, dtype=torch.long)
        self._step_ratio = 1

    def _alpha_prod(self, timestep: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if timestep < 0:
            return self.one.to(device=device, dtype=dtype)
        return self.alphas_cumprod[timestep].to(device=device, dtype=dtype)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alphas = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        sqrt_alpha_prod = alphas[timesteps].sqrt().view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = (1.0 - alphas[timesteps]).sqrt().view(-1, 1, 1)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def set_timesteps(self, num_inference_timesteps: int, device: torch.device | None = None) -> None:
        num_inference_timesteps = int(num_inference_timesteps)
        self._step_ratio = max(self.num_train_timesteps // num_inference_timesteps, 1)
        timesteps = torch.arange(
            self.num_train_timesteps - 1,
            -1,
            -self._step_ratio,
            dtype=torch.long,
        )[:num_inference_timesteps]
        self.timesteps = timesteps if device is None else timesteps.to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | int,
        sample: torch.Tensor,
    ) -> DDPMSchedulerOutput:
        t = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        prev_t = t - self._step_ratio
        alpha_prod_t = self._alpha_prod(t, sample.device, sample.dtype)
        alpha_prod_t_prev = self._alpha_prod(prev_t, sample.device, sample.dtype)
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

        pred_original_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_coeff * pred_original_sample + current_sample_coeff * sample
        if t > 0:
            variance = ((beta_prod_t_prev / beta_prod_t) * current_beta_t).clamp(min=1e-20)
            pred_prev_sample = pred_prev_sample + variance.sqrt() * torch.randn_like(sample)
        return DDPMSchedulerOutput(prev_sample=pred_prev_sample)


class DiffusionUNetActionHead(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        action_dim: int,
        observation_horizon: int = 2,
        action_horizon: int = 8,
        prediction_horizon: int = 16,
        diffusion_step_embed_dim: int = 256,
        down_dims: list[int] | tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        ddpm_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.action_dim = action_dim
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.prediction_horizon = prediction_horizon
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        ddpm_cfg = dict(ddpm_cfg or {})
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(ddpm_cfg.get("num_train_timesteps", 100)),
            beta_schedule=ddpm_cfg.get("beta_schedule", "squaredcos_cap_v2"),
            clip_sample=bool(ddpm_cfg.get("clip_sample", True)),
            prediction_type=ddpm_cfg.get("prediction_type", "epsilon"),
        )
        self.num_inference_timesteps = int(ddpm_cfg.get("num_inference_timesteps", 100))

    @property
    def clip_sample(self) -> bool:
        return self.noise_scheduler.clip_sample

    @clip_sample.setter
    def clip_sample(self, value: bool) -> None:
        self.noise_scheduler.clip_sample = bool(value)

    def training_loss(self, obs_cond: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.shape[1] != self.prediction_horizon:
            raise ValueError(
                f"DiffusionUNetActionHead expected action prediction horizon {self.prediction_horizon}, "
                f"got {action.shape[1]}."
            )
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (action.shape[0],),
            device=action.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        return F.mse_loss(noise_pred, noise)

    def forward(self, obs_cond: torch.Tensor) -> torch.Tensor:
        noisy_action = torch.randn(
            (obs_cond.shape[0], self.prediction_horizon, self.action_dim),
            device=obs_cond.device,
            dtype=obs_cond.dtype,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=obs_cond.device)
        naction = noisy_action
        for timestep in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(naction, timestep, global_cond=obs_cond)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=naction,
            ).prev_sample

        start = self.observation_horizon - 1
        end = start + self.action_horizon
        return naction[:, start:end]


__all__ = [
    "ConditionalResidualBlock1D",
    "ConditionalUnet1D",
    "Conv1dBlock",
    "DDPMScheduler",
    "DDPMSchedulerOutput",
    "DiffusionUNetActionHead",
    "Downsample1d",
    "SinusoidalPosEmb",
    "Upsample1d",
]
