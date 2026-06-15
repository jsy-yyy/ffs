from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from .base import BaseAdapter
from .reshape_tokens import ReshapeTokensAdapter
from .spatial_query import SpatialQueryAdapter
from .spatial_softmax import SpatialSoftmaxAdapter
from .vector import VectorAdapter


ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    "reshape_tokens": ReshapeTokensAdapter,
    "spatial_softmax": SpatialSoftmaxAdapter,
    "spatial_query": SpatialQueryAdapter,
    "vector": VectorAdapter,
}


def register_adapter(name: str, cls: type[BaseAdapter]) -> None:
    if name in ADAPTER_REGISTRY:
        raise ValueError(f"Adapter '{name}' is already registered.")
    ADAPTER_REGISTRY[name] = cls


def _validate_config_kwargs(cls: type[BaseAdapter], cfg: Mapping[str, Any]) -> None:
    signature = inspect.signature(cls.__init__)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return
    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    context_keys = {"backbone", "policy_cfg", "dataset_cfg"}
    unknown = sorted(set(cfg) - allowed - context_keys)
    if unknown:
        raise TypeError(f"{cls.__name__} got unsupported config keys: {', '.join(unknown)}")


def build_adapter(
    cfg: Mapping[str, Any],
    *,
    backbone: nn.Module,
    policy_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
) -> BaseAdapter:
    cfg = dict(cfg or {})
    adapter_type = cfg.pop("type", None)
    if adapter_type is None:
        raise ValueError("adapter.type is required.")
    try:
        adapter_cls = ADAPTER_REGISTRY[str(adapter_type)]
    except KeyError as exc:
        available = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(f"Unknown adapter type '{adapter_type}'. Available: {available}") from exc

    _validate_config_kwargs(adapter_cls, cfg)
    return adapter_cls(backbone=backbone, policy_cfg=policy_cfg, dataset_cfg=dataset_cfg, **cfg)


__all__ = ["ADAPTER_REGISTRY", "build_adapter", "register_adapter"]
