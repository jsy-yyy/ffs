from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogisticNormal

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


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("RDT positional embeddings require an even hidden_size.")
    omega = torch.arange(embed_dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    out = pos.reshape(-1).to(torch.float64).unsqueeze(1) * omega.unsqueeze(0)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def _multimodal_pos_embed(embed_dim: int, mm_lens: OrderedDict[str, int]) -> torch.Tensor:
    total_len = sum(abs(length) for length in mm_lens.values())
    pos_emb = torch.zeros(total_len, embed_dim, dtype=torch.float32)
    modality_emb = None
    if len(mm_lens) > 1:
        modality_emb = _get_1d_sincos_pos_embed(
            embed_dim,
            torch.arange(len(mm_lens), dtype=torch.float32),
        )

    start = 0
    for idx, (_, length) in enumerate(mm_lens.items()):
        abs_len = abs(length)
        if length > 1:
            emb = _get_1d_sincos_pos_embed(
                embed_dim,
                torch.arange(length, dtype=torch.float32),
            )
        else:
            emb = torch.zeros(abs_len, embed_dim, dtype=torch.float32)
        if modality_emb is not None:
            emb = emb + modality_emb[idx]
        pos_emb[start : start + abs_len] = emb
        start += abs_len
    return pos_emb


def _get_2d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError("2D positional embeddings require hidden_size divisible by 4.")
    if pos.shape[-1] != 2:
        raise ValueError(f"Expected token positions [...,2], got {tuple(pos.shape)}.")
    half_dim = embed_dim // 2
    y_emb = _get_1d_sincos_pos_embed(half_dim, pos[..., 0].reshape(-1))
    x_emb = _get_1d_sincos_pos_embed(half_dim, pos[..., 1].reshape(-1))
    return torch.cat([y_emb, x_emb], dim=-1).view(*pos.shape[:-1], embed_dim)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb.to(dtype=self.dtype))


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None,
        norm_eps: float,
        use_flash_attn: bool,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        if hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.num_repeats = self.num_heads // self.num_kv_heads
        self.use_flash_attn = use_flash_attn
        self.attn_scale = 1.0 / math.sqrt(self.head_size)

        self.wq = nn.Linear(hidden_size, self.num_heads * self.head_size, bias=False)
        self.wkv = nn.Linear(hidden_size, self.num_kv_heads * self.head_size * 2, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_size, hidden_size, bias=False)
        self.norm_q = RMSNorm(self.head_size, eps=norm_eps)
        self.norm_k = RMSNorm(self.head_size, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        xq = self.wq(x).view(batch, seq_len, self.num_heads, self.head_size)
        xkv = self.wkv(x).view(batch, seq_len, self.num_kv_heads, self.head_size, 2)
        xk, xv = xkv.unbind(-1)

        xq = self.norm_q(xq)
        xk = self.norm_k(xk)
        xk = _repeat_kv(xk, self.num_repeats)
        xv = _repeat_kv(xv, self.num_repeats)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            out = torch.matmul(scores, xv)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None,
        norm_eps: float,
        use_flash_attn: bool,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        if hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.num_repeats = self.num_heads // self.num_kv_heads
        self.use_flash_attn = use_flash_attn
        self.attn_scale = 1.0 / math.sqrt(self.head_size)

        self.wq = nn.Linear(hidden_size, self.num_heads * self.head_size, bias=False)
        self.wkv = nn.Linear(hidden_size, self.num_kv_heads * self.head_size * 2, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_size, hidden_size, bias=False)
        self.norm_q = RMSNorm(self.head_size, eps=norm_eps)
        self.norm_k = RMSNorm(self.head_size, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        _, cond_len, _ = c.shape

        xq = self.wq(x).view(batch, seq_len, self.num_heads, self.head_size)
        ckv = self.wkv(c).view(batch, cond_len, self.num_kv_heads, self.head_size, 2)
        ck, cv = ckv.unbind(-1)

        xq = self.norm_q(xq)
        ck = self.norm_k(ck)
        ck = _repeat_kv(ck, self.num_repeats)
        cv = _repeat_kv(cv, self.num_repeats)

        xq = xq.transpose(1, 2)
        ck = ck.transpose(1, 2)
        cv = cv.transpose(1, 2)

        attn_mask = None
        if bias is not None:
            if bias.shape != (batch, cond_len):
                raise ValueError(f"CrossAttention bias expected shape {(batch, cond_len)}, got {tuple(bias.shape)}.")
            attn_mask = bias.to(device=x.device, dtype=xq.dtype).reshape(batch, 1, 1, cond_len)
            attn_mask = attn_mask.expand(-1, -1, seq_len, -1)
        if mask is not None:
            if mask.shape != (batch, cond_len):
                raise ValueError(f"CrossAttention mask expected shape {(batch, cond_len)}, got {tuple(mask.shape)}.")
            keep = mask.reshape(batch, 1, 1, cond_len).expand(-1, -1, seq_len, -1)
            if attn_mask is None:
                attn_mask = keep
            else:
                attn_mask = attn_mask.masked_fill(keep.logical_not(), float("-inf"))

        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                xq,
                ck,
                cv,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            scores = torch.matmul(xq, ck.transpose(2, 3)) * self.attn_scale
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(attn_mask.logical_not(), float("-inf"))
                else:
                    scores = scores + attn_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            out = torch.matmul(scores, cv)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RDTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None,
        norm_eps: float,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        use_flash_attn: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            norm_eps=norm_eps,
            use_flash_attn=use_flash_attn,
        )
        self.cross_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.cond_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.cross_attn = CrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            norm_eps=norm_eps,
            use_flash_attn=use_flash_attn,
        )
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn = FeedForward(
            dim=hidden_size,
            hidden_dim=4 * hidden_size,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
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
        condition_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t_state).chunk(9, dim=1)
        )

        h = x + gate_attn.unsqueeze(1) * self.attn(
            _modulate(self.attn_norm(x), shift_attn, scale_attn)
        )
        h = h + gate_cross.unsqueeze(1) * self.cross_attn(
            _modulate(self.cross_norm(h), shift_cross, scale_cross),
            c=self.cond_norm(condition),
            mask=condition_mask,
            bias=condition_bias,
        )
        return h + gate_mlp.unsqueeze(1) * self.ffn(
            _modulate(self.ffn_norm(h), shift_mlp, scale_mlp)
        )


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, output_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
        )

    def forward(self, x: torch.Tensor, t_state: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t_state).chunk(2, dim=1)
        return self.ffn(_modulate(self.ffn_norm(x), shift, scale))


def _build_adapter(projector_type: str, in_features: int, out_features: int) -> nn.Module:
    if projector_type == "linear":
        return nn.Linear(in_features, out_features)

    mlp_silu_match = re.match(r"^mlp(\d+)x_silu$", projector_type)
    if mlp_silu_match is not None:
        depth = int(mlp_silu_match.group(1))
        if depth < 1:
            raise ValueError(f"Invalid projector depth in '{projector_type}'.")
        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
        for _ in range(1, depth):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(out_features, out_features))
        return nn.Sequential(*layers)

    raise ValueError(f"Unknown projector type: {projector_type}")


class RDTActionHead(ActionHead):
    """RDT2-style action expert conditioned on FoundationStereo tokens."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        action_horizon: int,
        prediction_horizon: int | None = None,
        frame_token_dim: int | None = None,
        condition_len: int | None = None,
        num_history_frames: int | None = None,
        tokens_per_frame: int | None = None,
        hidden_size: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        norm_eps: float = 1e-5,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        use_flash_attn: bool = True,
        num_register_tokens: int = 4,
        num_inference_steps: int = 10,
        sample_init: Literal["randn", "zeros"] = "randn",
        clip_sample: bool = False,
        act_adaptor: str = "mlp3x_silu",
        state_adaptor: str = "mlp3x_silu",
        vision_adaptor: str = "linear",
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
        if num_kv_heads is not None and num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        if sample_init not in ("randn", "zeros"):
            raise ValueError("sample_init must be 'randn' or 'zeros'.")
        self.prediction_horizon = int(prediction_horizon or action_horizon)
        if self.prediction_horizon < action_horizon:
            raise ValueError("prediction_horizon must be >= action_horizon.")
            
        self.frame_token_dim = frame_token_dim
        self.condition_len = condition_len
        self.hidden_size = hidden_size
        self.num_register_tokens = num_register_tokens
        self.num_inference_steps = num_inference_steps
        self.sample_init = sample_init
        self.clip_sample = bool(clip_sample)
        self.param_dtype = _as_dtype(dtype)

        if tokens_per_frame is None and num_history_frames is not None:
            if condition_len % num_history_frames != 0:
                raise ValueError("condition_len must be divisible by num_history_frames.")
            tokens_per_frame = condition_len // num_history_frames
        if tokens_per_frame is not None and tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive.")
        self.num_history_frames = num_history_frames
        self.tokens_per_frame = tokens_per_frame

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=self.param_dtype)
        self.act_adaptor = _build_adapter(act_adaptor, action_dim, hidden_size)
        self.state_adaptor = _build_adapter(state_adaptor, frame_token_dim, hidden_size)
        self.vision_adaptor = _build_adapter(vision_adaptor, frame_token_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [
                RDTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    norm_eps=norm_eps,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            output_size=action_dim,
            norm_eps=norm_eps,
        )
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_size))
        self.x_pos_emb = nn.Parameter(
            _multimodal_pos_embed(
                hidden_size,
                OrderedDict(
                    [
                        ("action", self.prediction_horizon),
                        ("register", num_register_tokens),
                    ]
                ),
            ).unsqueeze(0)
        )
        max_vision_len = max(condition_len - 1, 1)
        self.vision_pos_emb = nn.Parameter(
            _multimodal_pos_embed(
                hidden_size,
                OrderedDict([("vision", max_vision_len)]),
            ).unsqueeze(0)
        )
        self.state_pos_emb = nn.Parameter(
            _multimodal_pos_embed(hidden_size, OrderedDict([("state", 1)])).unsqueeze(0)
        )
        self.timestep_sampler = LogisticNormal(0, 1)

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
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.ffn[-1].weight)
        nn.init.zeros_(self.final_layer.ffn[-1].bias)

    def _state_token_indices(self, device: torch.device) -> torch.Tensor:
        if self.tokens_per_frame is None:
            return torch.tensor([self.condition_len - 1], device=device, dtype=torch.long)
        return torch.arange(
            self.tokens_per_frame - 1,
            self.condition_len,
            self.tokens_per_frame,
            device=device,
            dtype=torch.long,
        )

    def _initial_sample(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        shape = (frame_tokens.shape[0], self.prediction_horizon, self.action_dim)
        if self.sample_init == "zeros":
            return torch.zeros(shape, device=frame_tokens.device, dtype=self.param_dtype)
        return torch.randn(shape, device=frame_tokens.device, dtype=self.param_dtype)

    def _prepare_conditions(
        self,
        frame_tokens: torch.Tensor,
        condition_bias: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if frame_tokens.shape[1] != self.condition_len:
            raise ValueError(
                f"RDTActionHead expected {self.condition_len} condition tokens, got {frame_tokens.shape[1]}."
            )
        if frame_tokens.shape[2] != self.frame_token_dim:
            raise ValueError(
                f"RDTActionHead expected condition token dim {self.frame_token_dim}, got {frame_tokens.shape[2]}."
            )

        tokens = frame_tokens.to(self.param_dtype)
        state_indices = self._state_token_indices(tokens.device)
        state_tokens = tokens.index_select(1, state_indices)
        state_token = state_tokens[:, -1:, :]

        cond_mask = torch.ones(tokens.shape[1], device=tokens.device, dtype=torch.bool)
        cond_mask[state_indices] = False
        vision_tokens = tokens[:, cond_mask, :]
        if vision_tokens.shape[1] == 0:
            raise ValueError("RDTActionHead needs at least one non-state condition token.")
        vision_positions = None
        if token_positions is not None:
            if token_positions.shape != (tokens.shape[0], self.condition_len, 2):
                raise ValueError(
                    "RDTActionHead token_positions expected shape "
                    f"{(tokens.shape[0], self.condition_len, 2)}, got {tuple(token_positions.shape)}."
                )
            vision_positions = token_positions.to(device=tokens.device, dtype=self.param_dtype)[:, cond_mask, :]
        vision_bias = None
        if condition_bias is not None:
            if condition_bias.shape != (tokens.shape[0], self.condition_len):
                raise ValueError(
                    "RDTActionHead condition_bias expected shape "
                    f"{(tokens.shape[0], self.condition_len)}, got {tuple(condition_bias.shape)}."
                )
            vision_bias = condition_bias.to(device=tokens.device, dtype=self.param_dtype)[:, cond_mask]

        vision_cond = self.vision_adaptor(vision_tokens)
        if vision_positions is None:
            vision_cond = vision_cond + self.vision_pos_emb[:, : vision_cond.shape[1]]
        else:
            vision_cond = vision_cond + _get_2d_sincos_pos_embed(
                self.hidden_size,
                vision_positions,
            ).to(device=tokens.device, dtype=self.param_dtype)
        state_cond = self.state_adaptor(state_token) + self.state_pos_emb
        vision_mask = torch.ones(
            (tokens.shape[0], vision_cond.shape[1]),
            device=tokens.device,
            dtype=torch.bool,
        )
        return vision_cond, state_cond, vision_mask, vision_bias

    def _predict_velocity(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        vision_cond: torch.Tensor,
        state_cond: torch.Tensor,
        vision_mask: torch.Tensor,
        vision_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.act_adaptor(noisy_action.to(self.param_dtype))
        t = self.t_embedder(timestep.to(device=noisy_action.device))
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1)

        t_state = torch.cat([t.unsqueeze(1), state_cond], dim=1).reshape(x.shape[0], self.hidden_size * 2)
        r = self.register_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat([x, r], dim=1) + self.x_pos_emb

        for block in self.blocks:
            x = block(
                x=x,
                t_state=t_state,
                condition=vision_cond,
                condition_mask=vision_mask,
                condition_bias=vision_bias,
            )
        x = self.final_layer(x, t_state)
        return x[:, : self.prediction_horizon]

    def training_loss(
        self,
        frame_tokens: torch.Tensor,
        action: torch.Tensor,
        condition_bias: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action.shape[1] != self.prediction_horizon:
            raise ValueError(
                f"RDTActionHead expected action prediction horizon {self.prediction_horizon}, "
                f"got {action.shape[1]}."
            )
        vision_cond, state_cond, vision_mask, vision_bias = self._prepare_conditions(
            frame_tokens,
            condition_bias,
            token_positions,
        )
        action = action.to(self.param_dtype)
        noise = torch.randn_like(action)
        timesteps = self.timestep_sampler.sample((action.shape[0],))[:, 0].to(
            device=action.device,
            dtype=action.dtype,
        )
        noisy_action = action * timesteps.view(-1, 1, 1) + noise * (1 - timesteps.view(-1, 1, 1))
        pred = self._predict_velocity(
            noisy_action=noisy_action,
            timestep=timesteps,
            vision_cond=vision_cond,
            state_cond=state_cond,
            vision_mask=vision_mask,
            vision_bias=vision_bias,
        )
        return F.mse_loss(pred, action - noise)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        condition_bias: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vision_cond, state_cond, vision_mask, vision_bias = self._prepare_conditions(
            frame_tokens,
            condition_bias,
            token_positions,
        )
        noisy_action = self._initial_sample(frame_tokens)
        timestep = torch.tensor([0.0], device=frame_tokens.device, dtype=self.param_dtype)
        step_size = 1.0 / self.num_inference_steps

        for _ in range(self.num_inference_steps):
            pred = self._predict_velocity(
                noisy_action=noisy_action,
                timestep=timestep,
                vision_cond=vision_cond,
                state_cond=state_cond,
                vision_mask=vision_mask,
                vision_bias=vision_bias,
            )
            noisy_action = noisy_action + pred * step_size
            if self.clip_sample:
                noisy_action = noisy_action.clamp(-1.0, 1.0)
            timestep = timestep + step_size

        start = (self.num_history_frames or 1) - 1
        end = start + self.action_horizon
        return noisy_action[:, start:end].to(frame_tokens.dtype)
