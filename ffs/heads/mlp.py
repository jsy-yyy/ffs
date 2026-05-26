from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ActionHead


class ResidualMLPBlock(nn.Module):
    """Pre-norm hidden residual block for the MLP action head."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class MLPActionHead(ActionHead):
    """Maps conditional tokens to an action chunk with residual MLP blocks."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_dim=input_dim, action_dim=action_dim, action_horizon=action_horizon)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim, dropout=dropout)
            for _ in range(max(num_blocks, 0))
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_horizon * action_dim)

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        b = frame_tokens.shape[0]
        # frame_tokens: [B, N, C] -> action: [B, action_horizon, action_dim]
        hidden = self.input_proj(frame_tokens.flatten(1))
        for block in self.blocks:
            hidden = block(hidden)
        action = self.out_proj(self.final_norm(hidden))
        return action.view(b, self.action_horizon, self.action_dim)

    def training_loss(self, frame_tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(frame_tokens), action)
