from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ffs.config import checkpoint_config_candidates


@dataclass
class PolicyConfig:
    host: str = "127.0.0.1"
    port: int = 29068
    ffs_root: str = "/data/jsy/ffs"
    checkpoint: str = "outputs/robomimic_square_diffusion_aligned/latest.pt"
    config_path: str | None = None
    device: str = "cuda:0"
    amp: bool = True
    use_ema: bool = True
    sample_init: str | None = "zeros"
    clip_sample: bool | None = None
    disparity_ablation: str = "none"
    debug_actions: bool = False


@dataclass
class EnvConfig:
    dataset_path: str = "/data/jsy/robomimic/datasets/square/ph/stereo_image_v15.hdf5"
    n_rollouts: int = 50
    horizon: int = 400
    seed: int = 0
    camera_height: int | None = 224
    camera_width: int | None = 224
    stereo_baseline: float | None = 0.06
    mujoco_gl: str | None = "egl"
    render_gpu_device_id: int | None = None
    numba_disable_jit: bool = True


@dataclass
class ActionConfig:
    # null means execute the whole predicted FFS action horizon.
    execute_chunk_steps: int | None = None


@dataclass
class RecordConfig:
    save_metrics: bool = True
    save_episode_jsonl: bool = True
    save_video: bool = False
    video_skip: int = 5
    save_action_jsonl: bool = True


@dataclass
class EvalConfig:
    save_root: str = "outputs/robomimic_square_eval"
    dry_run: bool = False
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    record: RecordConfig = field(default_factory=RecordConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_eval_config(path: str | Path, overrides: dict[str, Any] | None = None) -> EvalConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = _merge_dict(data, overrides)

    policy = PolicyConfig(**data.pop("policy", {}))
    env = EnvConfig(**data.pop("env", {}))
    action = ActionConfig(**data.pop("action", {}))
    record = RecordConfig(**data.pop("record", {}))
    return EvalConfig(**data, policy=policy, env=env, action=action, record=record)


def resolve_ffs_config_path(checkpoint: str | Path, explicit_config: str | Path | None = None) -> Path:
    if explicit_config is not None:
        return Path(explicit_config)

    checkpoint_path = Path(checkpoint)
    candidates = checkpoint_config_candidates(checkpoint_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve FFS config for robomimic eval client. "
        f"Searched sidecar configs: {searched}. Pass policy.config_path explicitly."
    )


def load_ffs_sidecar_config(checkpoint: str | Path, explicit_config: str | Path | None = None) -> dict[str, Any]:
    path = resolve_ffs_config_path(checkpoint, explicit_config)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = [
    "ActionConfig",
    "EnvConfig",
    "EvalConfig",
    "PolicyConfig",
    "RecordConfig",
    "load_eval_config",
    "load_ffs_sidecar_config",
    "resolve_ffs_config_path",
]
