from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ActionHead(nn.Module, ABC):
    """Base class for action heads that map frame tokens to action chunks."""

    def __init__(self, input_dim: int, action_dim: int, action_horizon: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon

    @abstractmethod
    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        """frame_tokens: [B, N, C] -> action: [B, action_horizon, action_dim]."""

    @abstractmethod
    def training_loss(self, frame_tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the head-specific training loss for an action chunk."""
