from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ActionHead


def _as_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if not hasattr(torch, dtype):
        raise ValueError(f"Unknown torch dtype '{dtype}'.")
    value = getattr(torch, dtype)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"'{dtype}' is not a torch dtype.")
    return value


def _sincos_pos_embed(length: int, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32) / max(half, 1)
    )
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    emb = torch.cat([torch.sin(pos * freqs), torch.cos(pos * freqs)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FlowMatchScheduler:
    """Small inference scheduler matching the RDT flow-matching update."""

    def __init__(
        self,
        num_inference_steps: int = 10,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = True,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int) -> None:
        sigma_start = self.sigma_max
        if self.extra_one_step:
            sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, to_final: bool) -> torch.Tensor:
        timestep_value = timestep.flatten()[0].detach().cpu()
        timestep_id = torch.argmin((self.timesteps - timestep_value).abs())
        sigma = self.sigmas[timestep_id].to(sample.device, sample.dtype)
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = torch.zeros((), device=sample.device, dtype=sample.dtype)
        else:
            sigma_next = self.sigmas[timestep_id + 1].to(sample.device, sample.dtype)
        return sample + model_output * (sigma_next - sigma)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb.to(next(self.mlp.parameters()).dtype))


class RDTBlock(nn.Module):
    """RDT-style block with adaLN, self-attention, and cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.self_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.cross_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.cond_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 9),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_state: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t_state).chunk(9, dim=1)
        )

        self_in = _modulate(self.self_norm(x), shift_attn, scale_attn)
        self_out = self.self_attn(self_in, self_in, self_in, need_weights=False)[0]
        x = x + gate_attn.unsqueeze(1) * self_out

        key_padding_mask = None
        if condition_mask is not None:
            key_padding_mask = ~condition_mask
        cross_in = _modulate(self.cross_norm(x), shift_cross, scale_cross)
        cond = self.cond_norm(condition)
        cross_out = self.cross_attn(
            cross_in,
            cond,
            cond,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + gate_cross.unsqueeze(1) * cross_out

        ffn_in = _modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.ffn(ffn_in)


class RDTActionHead(ActionHead):
    """RDT-style denoising action head conditioned on stereo tokens.

    This follows the open-p2p RDT head pattern: action tokens are iteratively
    denoised with flow matching while cross-attending to projected condition
    tokens and using timestep/state conditioning through adaLN.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        action_horizon: int,
        frame_token_dim: int | None = None,
        condition_len: int | None = None,
        hidden_size: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        num_register_tokens: int = 4,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 10,
        flow_match_shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = True,
        sample_init: Literal["randn", "zeros"] = "randn",
        dtype: str | torch.dtype = torch.float32,
    ) -> None:
        super().__init__(input_dim=input_dim, action_dim=action_dim, action_horizon=action_horizon)
        if condition_len is None:
            condition_len = 1
        if frame_token_dim is None:
            if input_dim % condition_len != 0:
                raise ValueError(
                    "RDTActionHead needs frame_token_dim when input_dim is not divisible by condition_len."
                )
            frame_token_dim = input_dim // condition_len
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if sample_init not in ("randn", "zeros"):
            raise ValueError("sample_init must be 'randn' or 'zeros'.")

        self.frame_token_dim = frame_token_dim
        self.condition_len = condition_len
        self.hidden_size = hidden_size
        self.num_register_tokens = num_register_tokens
        self.sample_init = sample_init
        self.param_dtype = _as_dtype(dtype)

        self.condition_proj = nn.Linear(frame_token_dim, hidden_size)
        self.state_proj = nn.Linear(frame_token_dim, hidden_size)
        self.action_embedder = nn.Linear(action_dim, hidden_size)
        self.action_decoder = nn.Linear(hidden_size, action_dim)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                RDTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_size))
        self.x_pos_emb = nn.Parameter(
            _sincos_pos_embed(action_horizon + num_register_tokens, hidden_size).unsqueeze(0)
        )
        self.condition_pos_emb = nn.Parameter(
            _sincos_pos_embed(condition_len, hidden_size).unsqueeze(0)
        )
        self.scheduler = FlowMatchScheduler(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=flow_match_shift,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
        )

        self.initialize_weights()
        self.to(self.param_dtype)

    def initialize_weights(self) -> None:
        def init_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_module)
        nn.init.normal_(self.register_tokens, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def _initial_sample(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        shape = (frame_tokens.shape[0], self.action_horizon, self.action_dim)
        if self.sample_init == "zeros":
            return torch.zeros(shape, device=frame_tokens.device, dtype=self.param_dtype)
        return torch.randn(shape, device=frame_tokens.device, dtype=self.param_dtype)

    def _prepare_condition(self, frame_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_tokens.shape[1] != self.condition_len:
            raise ValueError(
                f"RDTActionHead expected {self.condition_len} condition tokens, got {frame_tokens.shape[1]}."
            )
        if frame_tokens.shape[2] != self.frame_token_dim:
            raise ValueError(
                f"RDTActionHead expected condition token dim {self.frame_token_dim}, got {frame_tokens.shape[2]}."
            )

        tokens = frame_tokens.to(self.param_dtype)
        condition = self.condition_proj(tokens) + self.condition_pos_emb[:, : tokens.shape[1]]
        state_token = self.state_proj(tokens[:, -1:, :])
        return condition, state_token

    def _predict_denoising_target(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        state_token: torch.Tensor,
    ) -> torch.Tensor:
        b = sample.shape[0]
        x = self.action_embedder(sample)
        r = self.register_tokens.expand(b, -1, -1)
        x = torch.cat([x, r], dim=1) + self.x_pos_emb

        t = self.t_embedder(timestep)
        t_state = torch.cat([t, state_token.squeeze(1)], dim=1)
        for block in self.blocks:
            x = block(x=x, t_state=t_state, condition=condition)
        x = x[:, : self.action_horizon]
        return self.action_decoder(x)

    def training_loss(self, frame_tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        condition, state_token = self._prepare_condition(frame_tokens)
        action = action.to(self.param_dtype)
        noise = torch.randn_like(action)

        raw_sigma = torch.rand(action.shape[0], device=action.device, dtype=action.dtype)
        raw_sigma = self.scheduler.sigma_min + raw_sigma * (self.scheduler.sigma_max - self.scheduler.sigma_min)
        sigma = self.scheduler.shift * raw_sigma / (1 + (self.scheduler.shift - 1) * raw_sigma)
        timestep = (sigma * self.scheduler.num_train_timesteps).to(torch.float32)
        sigma = sigma.view(-1, 1, 1)

        noisy_action = sigma * noise + (1 - sigma) * action
        target = noise - action
        pred = self._predict_denoising_target(
            sample=noisy_action,
            timestep=timestep,
            condition=condition,
            state_token=state_token,
        )
        return F.mse_loss(pred, target)

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        condition, state_token = self._prepare_condition(frame_tokens)
        sample = self._initial_sample(frame_tokens)
        timesteps = self.scheduler.timesteps.to(frame_tokens.device)

        for step_idx, timestep in enumerate(timesteps):
            timestep_batch = torch.full(
                (frame_tokens.shape[0],),
                float(timestep.item()),
                dtype=torch.float32,
                device=frame_tokens.device,
            )
            model_output = self._predict_denoising_target(
                sample=sample,
                timestep=timestep_batch,
                condition=condition,
                state_token=state_token,
            )
            sample = self.scheduler.step(
                model_output=model_output,
                timestep=timestep_batch,
                sample=sample,
                to_final=(step_idx + 1 == len(timesteps)),
            )
        return sample.to(frame_tokens.dtype)
