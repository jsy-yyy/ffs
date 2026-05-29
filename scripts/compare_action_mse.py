from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ffs.argparse_compat import add_boolean_optional_argument
from ffs import load_config_for_checkpoint
from ffs.datasets.lerobot import relative_action_to_absolute_eef_pose
from scripts.eval_offline import autocast_context, load_model, make_loader


DEFAULT_MLP_CHECKPOINT = "outputs/mlp_seed0_overfit_500/latest.pt"
DEFAULT_RDT_CHECKPOINT = "outputs_abseef/rdt_seed0_overfit_500_3e-5/latest.pt"


def parse_episodes(value: str) -> list[int]:
    episodes = [part.strip() for part in value.split(",")]
    if not episodes or any(part == "" for part in episodes):
        raise argparse.ArgumentTypeError("--episodes must be a comma-separated list of integers")
    try:
        return [int(part) for part in episodes]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--episodes must be a comma-separated list of integers") from exc


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_eval_config(
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    dataset_root: str | None,
    episodes: list[int],
) -> tuple[dict[str, Any], str]:
    cfg, config_source = load_config_for_checkpoint(checkpoint_path, config_path)
    if dataset_root:
        cfg["dataset"]["root"] = dataset_root
    cfg["dataset"]["episode_indices"] = episodes
    return cfg, config_source


def evaluate_model(
    *,
    name: str,
    config_path: str | Path | None,
    checkpoint_path: str | Path,
    dataset_root: str | None,
    episodes: list[int],
    device: torch.device,
    batch_size: int | None,
    num_workers: int | None,
    max_batches: int | None,
    log_interval: int,
    seed: int | None,
    sample_init: str | None,
    amp: bool,
) -> dict[str, float | int | str | list[int] | None]:
    set_seed(seed)
    cfg, config_source = load_eval_config(checkpoint_path, config_path, dataset_root, episodes)
    loader = make_loader(cfg, batch_size, num_workers)
    model = load_model(cfg, checkpoint_path, device)
    model.eval()

    if sample_init is not None and hasattr(model.action_head, "sample_init"):
        model.action_head.sample_init = sample_init

    use_amp = bool(amp) and device.type == "cuda"
    total_elements = 0
    total_samples = 0
    normalized_mse_sum = 0.0
    normalized_mae_sum = 0.0
    relative_mse_sum = 0.0
    relative_mae_sum = 0.0
    absolute_mse_sum = 0.0
    absolute_mae_sum = 0.0
    raw_mse_sum = 0.0
    raw_mae_sum = 0.0
    dataset = loader.dataset
    action_mode = getattr(dataset, "action_mode", None)
    is_robotwin = action_mode == "relative-eef"

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            if is_robotwin:
                relative_action = batch["relative_action"].to(device, non_blocking=True)
                absolute_action = batch["absolute_action"].to(device, non_blocking=True)
            else:
                raw_action = batch["raw_action"].to(device, non_blocking=True)

            with autocast_context(device, use_amp):
                pred = model(left, right, state)
                normalized_mse = F.mse_loss(pred, action, reduction="sum")
                normalized_mae = F.l1_loss(pred, action, reduction="sum")
                pred_denorm = dataset.denormalize_action(pred.float())
                if is_robotwin:
                    relative_mse = F.mse_loss(pred_denorm, relative_action.float(), reduction="sum")
                    relative_mae = F.l1_loss(pred_denorm, relative_action.float(), reduction="sum")
                    absolute_pred = relative_action_to_absolute_eef_pose(pred_denorm, state[:, -1].float())
                    absolute_mse = F.mse_loss(absolute_pred, absolute_action.float(), reduction="sum")
                    absolute_mae = F.l1_loss(absolute_pred, absolute_action.float(), reduction="sum")
                else:
                    raw_mse = F.mse_loss(pred_denorm, raw_action.float(), reduction="sum")
                    raw_mae = F.l1_loss(pred_denorm, raw_action.float(), reduction="sum")

            total_elements += action.numel()
            total_samples += action.shape[0]
            normalized_mse_sum += float(normalized_mse.detach().cpu())
            normalized_mae_sum += float(normalized_mae.detach().cpu())
            if is_robotwin:
                relative_mse_sum += float(relative_mse.detach().cpu())
                relative_mae_sum += float(relative_mae.detach().cpu())
                absolute_mse_sum += float(absolute_mse.detach().cpu())
                absolute_mae_sum += float(absolute_mae.detach().cpu())
            else:
                raw_mse_sum += float(raw_mse.detach().cpu())
                raw_mae_sum += float(raw_mae.detach().cpu())

            if log_interval and (batch_idx + 1) % log_interval == 0:
                running_normalized_mse = normalized_mse_sum / max(total_elements, 1)
                if is_robotwin:
                    running_relative_mse = relative_mse_sum / max(total_elements, 1)
                    running_absolute_mse = absolute_mse_sum / max(total_elements, 1)
                    print(
                        f"{name}: batch={batch_idx + 1} samples={total_samples} "
                        f"normalized_mse={running_normalized_mse:.8f} "
                        f"relative_mse={running_relative_mse:.8f} "
                        f"absolute_mse={running_absolute_mse:.8f}",
                        flush=True,
                    )
                else:
                    running_raw_mse = raw_mse_sum / max(total_elements, 1)
                    print(
                        f"{name}: batch={batch_idx + 1} samples={total_samples} "
                        f"normalized_mse={running_normalized_mse:.8f} "
                        f"raw_mse={running_raw_mse:.8f}",
                        flush=True,
                    )

    if total_elements == 0:
        raise RuntimeError(f"No evaluation batches were processed for {name}.")

    normalized_mse = normalized_mse_sum / total_elements
    metrics = {
        "config": str(config_path) if config_path is not None else None,
        "config_source": config_source,
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(cfg["dataset"]["root"]),
        "episodes": list(episodes),
        "samples": total_samples,
        "action_elements": total_elements,
        "sample_init": getattr(model.action_head, "sample_init", None),
        "mse": normalized_mse,
        "rmse": normalized_mse**0.5,
        "mae": normalized_mae_sum / total_elements,
        "normalized_mse": normalized_mse,
        "normalized_rmse": normalized_mse**0.5,
        "normalized_mae": normalized_mae_sum / total_elements,
    }
    if is_robotwin:
        relative_mse = relative_mse_sum / total_elements
        absolute_mse = absolute_mse_sum / total_elements
        metrics.update(
            {
                "relative_mse": relative_mse,
                "relative_rmse": relative_mse**0.5,
                "relative_mae": relative_mae_sum / total_elements,
                "absolute_mse": absolute_mse,
                "absolute_rmse": absolute_mse**0.5,
                "absolute_mae": absolute_mae_sum / total_elements,
            }
        )
    else:
        raw_mse = raw_mse_sum / total_elements
        metrics.update(
            {
                "raw_mse": raw_mse,
                "raw_rmse": raw_mse**0.5,
                "raw_mae": raw_mae_sum / total_elements,
            }
        )
    return metrics


def compare(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    if args.suppress_dynamo_errors:
        dynamo = importlib.import_module("torch._dynamo")
        dynamo.config.suppress_errors = True

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    mlp = evaluate_model(
        name="mlp",
        config_path=args.mlp_config,
        checkpoint_path=args.mlp_checkpoint,
        dataset_root=args.dataset_root,
        episodes=args.episodes,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        log_interval=args.log_interval,
        seed=args.seed,
        sample_init=None,
        amp=args.amp,
    )
    rdt = evaluate_model(
        name="rdt",
        config_path=args.rdt_config,
        checkpoint_path=args.rdt_checkpoint,
        dataset_root=args.dataset_root,
        episodes=args.episodes,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        log_interval=args.log_interval,
        seed=args.seed,
        sample_init=args.rdt_sample_init,
        amp=args.amp,
    )

    mlp_mse = float(mlp["mse"])
    rdt_mse = float(rdt["mse"])
    result = {
        "mlp": mlp,
        "rdt": rdt,
        "delta_mse": rdt_mse - mlp_mse,
        "ratio_mse": rdt_mse / mlp_mse if mlp_mse != 0 else None,
    }
    if "relative_mse" in mlp and "relative_mse" in rdt:
        mlp_relative_mse = float(mlp["relative_mse"])
        rdt_relative_mse = float(rdt["relative_mse"])
        mlp_absolute_mse = float(mlp["absolute_mse"])
        rdt_absolute_mse = float(rdt["absolute_mse"])
        result.update(
            {
                "relative_delta_mse": rdt_relative_mse - mlp_relative_mse,
                "relative_ratio_mse": rdt_relative_mse / mlp_relative_mse if mlp_relative_mse != 0 else None,
                "absolute_delta_mse": rdt_absolute_mse - mlp_absolute_mse,
                "absolute_ratio_mse": rdt_absolute_mse / mlp_absolute_mse if mlp_absolute_mse != 0 else None,
            }
        )
    if "raw_mse" in mlp and "raw_mse" in rdt:
        mlp_raw_mse = float(mlp["raw_mse"])
        rdt_raw_mse = float(rdt["raw_mse"])
        result.update(
            {
                "raw_delta_mse": rdt_raw_mse - mlp_raw_mse,
                "raw_ratio_mse": rdt_raw_mse / mlp_raw_mse if mlp_raw_mse != 0 else None,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLP and RDT action prediction MSE offline.")
    parser.add_argument("--mlp-config", default=None)
    parser.add_argument("--mlp-checkpoint", default=DEFAULT_MLP_CHECKPOINT)
    parser.add_argument("--rdt-config", default=None)
    parser.add_argument("--rdt-checkpoint", default=DEFAULT_RDT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episodes", type=parse_episodes, default=parse_episodes("0"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rdt-sample-init", choices=["randn", "zeros"], default=None)
    add_boolean_optional_argument(parser, "--amp", default=True)
    add_boolean_optional_argument(parser, "--suppress-dynamo-errors", default=True)
    args = parser.parse_args()

    metrics = compare(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
