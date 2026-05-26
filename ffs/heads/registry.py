from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from .base import ActionHead
from .mlp import MLPActionHead
from .rdt import RDTActionHead


ACTION_HEAD_REGISTRY: dict[str, type[ActionHead]] = {
    "mlp": MLPActionHead,
    "rdt": RDTActionHead,
}


def register_action_head(name: str, cls: type[ActionHead]) -> None:
    if name in ACTION_HEAD_REGISTRY:
        raise ValueError(f"Action head '{name}' is already registered.")
    ACTION_HEAD_REGISTRY[name] = cls


def _filter_kwargs(cls: type[ActionHead], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    parameters = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return kwargs
    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


def _validate_config_kwargs(cls: type[ActionHead], cfg: Mapping[str, Any]) -> None:
    signature = inspect.signature(cls.__init__)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return
    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise TypeError(f"{cls.__name__} got unsupported config keys: {', '.join(unknown)}")


def build_action_head(
    cfg: Mapping[str, Any] | None,
    *,
    input_dim: int,
    action_dim: int,
    action_horizon: int,
    **build_context: Any,
) -> ActionHead:
    cfg = dict(cfg or {})
    head_type = cfg.pop("type", "mlp")
    try:
        head_cls = ACTION_HEAD_REGISTRY[head_type]
    except KeyError as exc:
        available = ", ".join(sorted(ACTION_HEAD_REGISTRY))
        raise ValueError(f"Unknown action head type '{head_type}'. Available: {available}") from exc

    _validate_config_kwargs(head_cls, cfg)
    kwargs = {
        "input_dim": input_dim,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        **build_context,
        **cfg,
    }
    return head_cls(**_filter_kwargs(head_cls, kwargs))
