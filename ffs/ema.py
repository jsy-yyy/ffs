from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EMAConfig:
    power: float = 0.75
    inv_gamma: float = 1.0
    max_decay: float = 0.9999

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> EMAConfig:
        cfg = dict(cfg or {})
        config = cls(
            power=float(cfg.get("power", cls.power)),
            inv_gamma=float(cfg.get("inv_gamma", cls.inv_gamma)),
            max_decay=float(cfg.get("max_decay", cls.max_decay)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.power <= 0:
            raise ValueError("EMA power must be positive.")
        if self.inv_gamma <= 0:
            raise ValueError("EMA inv_gamma must be positive.")
        if not 0.0 < self.max_decay < 1.0:
            raise ValueError("EMA max_decay must be in (0, 1).")

    def as_dict(self) -> dict[str, float]:
        return {
            "power": self.power,
            "inv_gamma": self.inv_gamma,
            "max_decay": self.max_decay,
        }


class ModelEMA:
    """Exponential moving average for trainable model parameters."""

    def __init__(self, model: nn.Module, config: EMAConfig, step: int = 0) -> None:
        self.config = config
        self.step = int(step)
        self.shadow: dict[str, torch.Tensor] = {}
        self.reset(model, step=step)

    def reset(self, model: nn.Module, step: int = 0) -> None:
        self.step = int(step)
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().float().clone()

    def decay_for_step(self, step: int | None = None) -> float:
        step = self.step if step is None else int(step)
        if step <= 0:
            return 0.0
        decay = 1.0 - (1.0 + step / self.config.inv_gamma) ** (-self.config.power)
        return min(self.config.max_decay, decay)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        decay = self.decay_for_step()
        current = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            param = current.get(name)
            if param is None:
                continue
            shadow.mul_(decay).add_(param.detach().float(), alpha=1.0 - decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        missing = sorted(set(self.shadow) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(self.shadow))
        for name in sorted(set(self.shadow) & set(state_dict)):
            target = self.shadow[name]
            value = state_dict[name]
            if target.shape != value.shape:
                raise ValueError(
                    f"EMA parameter {name!r} has shape {tuple(value.shape)}, "
                    f"expected {tuple(target.shape)}."
                )
            self.shadow[name] = value.detach().to(device=target.device, dtype=torch.float32).clone()
        return missing, unexpected


@torch.no_grad()
def load_ema_state_dict(model: nn.Module, ema_state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    model_state = model.state_dict()
    missing = sorted(set(ema_state) - set(model_state))
    loaded: list[str] = []
    for name, value in ema_state.items():
        target = model_state.get(name)
        if target is None:
            continue
        if target.shape != value.shape:
            raise ValueError(
                f"EMA parameter {name!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(target.shape)}."
            )
        target.copy_(value.to(device=target.device, dtype=target.dtype))
        loaded.append(name)
    return loaded, missing


def ema_config_from_train_cfg(train_cfg: dict[str, Any]) -> EMAConfig | None:
    ema_cfg = train_cfg.get("ema")
    if not isinstance(ema_cfg, dict) or not bool(ema_cfg.get("enabled", False)):
        return None
    return EMAConfig.from_dict(ema_cfg)


__all__ = [
    "EMAConfig",
    "ModelEMA",
    "ema_config_from_train_cfg",
    "load_ema_state_dict",
]
