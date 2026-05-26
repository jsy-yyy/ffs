from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"The '(repr|frozen)' attribute with value .* was provided to the `Field\(\)` function",
    module=r"pydantic\._internal\._generate_schema",
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ffs import load_config_for_checkpoint
from ffs.datasets import LeRobotStereoDataset
from ffs.datasets.lerobot import relative_action_to_absolute_eef_pose
from ffs.policies.stereo_action_policy import build_policy
from scripts.train import autocast_context


def make_loader(cfg: dict[str, Any], batch_size: int | None, num_workers: int | None) -> DataLoader:
    policy_cfg = cfg["policy"]
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg.get("train", {})
    dataset = LeRobotStereoDataset(
        root=dataset_cfg["root"],
        camera_pairs=dataset_cfg["camera_pairs"],
        num_history_frames=policy_cfg["num_history_frames"],
        action_horizon=policy_cfg["action_horizon"],
        image_size=dataset_cfg.get("image_size"),
        episode_indices=dataset_cfg.get("episode_indices"),
        action_normalization=dataset_cfg.get("action_normalization"),
    )
    if int(policy_cfg["state_dim"]) != dataset.state_dim:
        raise ValueError(
            f"policy.state_dim={policy_cfg['state_dim']} does not match dataset state_dim={dataset.state_dim}"
        )
    if int(policy_cfg["action_dim"]) != dataset.action_dim:
        raise ValueError(
            f"policy.action_dim={policy_cfg['action_dim']} does not match dataset action_dim={dataset.action_dim}"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size or int(train_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(num_workers if num_workers is not None else train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def load_model(cfg: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_policy(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model


def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    cfg, config_source = load_config_for_checkpoint(args.checkpoint, args.config)
    if args.dataset_root:
        cfg["dataset"]["root"] = args.dataset_root
    if args.suppress_dynamo_errors:
        dynamo = importlib.import_module("torch._dynamo")
        dynamo.config.suppress_errors = True

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    loader = make_loader(cfg, args.batch_size, args.num_workers)
    model = load_model(cfg, args.checkpoint, device)
    model.eval()

    if args.sample_init is not None and hasattr(model.action_head, "sample_init"):
        model.action_head.sample_init = args.sample_init

    use_amp = bool(args.amp) and device.type == "cuda"
    total_elements = 0
    total_samples = 0
    normalized_mse_sum = 0.0
    normalized_mae_sum = 0.0
    relative_mse_sum = 0.0
    relative_mae_sum = 0.0
    absolute_mse_sum = 0.0
    absolute_mae_sum = 0.0
    dataset = loader.dataset

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            relative_action = batch["relative_action"].to(device, non_blocking=True)
            absolute_action = batch["absolute_action"].to(device, non_blocking=True)

            with autocast_context(device, use_amp):
                pred = model(left, right, state)
                normalized_mse = F.mse_loss(pred, action, reduction="sum")
                normalized_mae = F.l1_loss(pred, action, reduction="sum")
                pred_relative = dataset.denormalize_action(pred.float())
                relative_mse = F.mse_loss(pred_relative, relative_action.float(), reduction="sum")
                relative_mae = F.l1_loss(pred_relative, relative_action.float(), reduction="sum")
                absolute_pred = relative_action_to_absolute_eef_pose(pred_relative, state[:, -1].float())
                absolute_mse = F.mse_loss(absolute_pred, absolute_action.float(), reduction="sum")
                absolute_mae = F.l1_loss(absolute_pred, absolute_action.float(), reduction="sum")

            total_elements += action.numel()
            total_samples += action.shape[0]
            normalized_mse_sum += float(normalized_mse.detach().cpu())
            normalized_mae_sum += float(normalized_mae.detach().cpu())
            relative_mse_sum += float(relative_mse.detach().cpu())
            relative_mae_sum += float(relative_mae.detach().cpu())
            absolute_mse_sum += float(absolute_mse.detach().cpu())
            absolute_mae_sum += float(absolute_mae.detach().cpu())

            if args.log_interval and (batch_idx + 1) % args.log_interval == 0:
                running_normalized_mse = normalized_mse_sum / max(total_elements, 1)
                running_relative_mse = relative_mse_sum / max(total_elements, 1)
                running_absolute_mse = absolute_mse_sum / max(total_elements, 1)
                print(
                    f"batch={batch_idx + 1} samples={total_samples} "
                    f"normalized_mse={running_normalized_mse:.8f} "
                    f"relative_mse={running_relative_mse:.8f} "
                    f"absolute_mse={running_absolute_mse:.8f}",
                    flush=True,
                )

    if total_elements == 0:
        raise RuntimeError("No evaluation batches were processed.")

    normalized_mse = normalized_mse_sum / total_elements
    relative_mse = relative_mse_sum / total_elements
    absolute_mse = absolute_mse_sum / total_elements
    return {
        "checkpoint": str(args.checkpoint),
        "config_source": config_source,
        "dataset_root": str(cfg["dataset"]["root"]),
        "samples": total_samples,
        "action_elements": total_elements,
        "mse": normalized_mse,
        "rmse": normalized_mse**0.5,
        "mae": normalized_mae_sum / total_elements,
        "normalized_mse": normalized_mse,
        "normalized_rmse": normalized_mse**0.5,
        "normalized_mae": normalized_mae_sum / total_elements,
        "relative_mse": relative_mse,
        "relative_rmse": relative_mse**0.5,
        "relative_mae": relative_mae_sum / total_elements,
        "absolute_mse": absolute_mse,
        "absolute_rmse": absolute_mse**0.5,
        "absolute_mae": absolute_mae_sum / total_elements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="outputs/rdt_version_1_clean/latest.pt")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-init", choices=["randn", "zeros"], default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suppress-dynamo-errors", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
