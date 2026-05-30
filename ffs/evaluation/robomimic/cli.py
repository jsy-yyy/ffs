from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ffs.argparse_compat import add_boolean_optional_argument

from .config import load_eval_config
from .runner import RobomimicEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FFS online evaluation in robomimic.")
    parser.add_argument("--config", default="configs/robomimic_square_eval.yaml")
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--n-rollouts", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--execute-chunk-steps", type=int, default=None)
    parser.add_argument("--mujoco-gl", default=None)
    parser.add_argument("--render-gpu-device-id", type=int, default=None)
    add_boolean_optional_argument(parser, "--save-video", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.save_root is not None:
        overrides["save_root"] = args.save_root
    if args.dry_run:
        overrides["dry_run"] = True

    policy: dict[str, Any] = {}
    for arg_name, cfg_name in {
        "checkpoint": "checkpoint",
        "config_path": "config_path",
        "host": "host",
        "port": "port",
    }.items():
        value = getattr(args, arg_name)
        if value is not None:
            policy[cfg_name] = value
    if policy:
        overrides["policy"] = policy

    env: dict[str, Any] = {}
    for arg_name, cfg_name in {
        "dataset": "dataset_path",
        "n_rollouts": "n_rollouts",
        "horizon": "horizon",
        "seed": "seed",
        "mujoco_gl": "mujoco_gl",
        "render_gpu_device_id": "render_gpu_device_id",
    }.items():
        value = getattr(args, arg_name)
        if value is not None:
            env[cfg_name] = value
    if env:
        overrides["env"] = env

    if args.execute_chunk_steps is not None:
        overrides["action"] = {"execute_chunk_steps": args.execute_chunk_steps}
    if args.save_video is not None:
        overrides["record"] = {"save_video": args.save_video}
    return overrides


def main() -> None:
    args = parse_args()
    config = load_eval_config(args.config, _overrides(args))
    config.save_root = str(Path(config.save_root).resolve())
    evaluator = RobomimicEvaluator(config)

    if config.dry_run:
        print(json.dumps(evaluator.dry_run_summary(), indent=2, ensure_ascii=False))
        return

    metrics = evaluator.run()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
