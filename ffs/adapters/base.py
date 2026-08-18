from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class AdapterOutput:
    cond: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    condition_bias: torch.Tensor | None = None
    token_positions: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    attention: dict[str, torch.Tensor] | None = None


class BaseAdapter(nn.Module):
    output_kind: Literal["cond", "tokens"]

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        backbone_out: dict[str, torch.Tensor],
        state: torch.Tensor,
        *,
        batch: int,
        time: int,
        views: int,
        return_attention: bool = False,
    ) -> AdapterOutput:
        raise NotImplementedError


def resolve_feature_names(cfg_value: object, backbone: nn.Module) -> tuple[str, ...]:
    if cfg_value is None or cfg_value == "auto":
        return tuple(getattr(backbone, "feature_names"))
    if isinstance(cfg_value, str):
        return (cfg_value,)
    return tuple(cfg_value)  # type: ignore[arg-type]


def feature_view_count(backbone: nn.Module, name: str, default_views: int) -> int:
    view_counts = getattr(backbone, "feature_view_counts", None)
    if isinstance(view_counts, dict) and name in view_counts:
        return int(view_counts[name])
    return int(default_views)
