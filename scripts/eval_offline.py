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

from ffs.argparse_compat import add_boolean_optional_argument
from ffs import load_config_for_checkpoint
from ffs.datasets import build_stereo_lerobot_dataset
from ffs.datasets.lerobot import relative_action_to_absolute_eef_pose
from ffs.policies.stereo_action_policy import build_policy
from ffs.visualization import (
    default_query_attention_config,
    render_query_attention_frames,
    save_query_attention_video,
)
from scripts.train import autocast_context


def make_loader(cfg: dict[str, Any], batch_size: int | None, num_workers: int | None) -> DataLoader:
    policy_cfg = cfg["policy"]
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg.get("train", {})
    dataset = build_stereo_lerobot_dataset(dataset_cfg, policy_cfg)
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


def resolve_query_attention_config(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    configured = cfg.get("visualization", {}).get("query_attention", {})
    query_cfg = default_query_attention_config(configured)
    if args.visualize_query_attention:
        query_cfg["enabled"] = True
    for key, attr in {
        "mode": "query_attention_mode",
        "sources": "query_attention_sources",
        "view": "query_attention_view",
        "time": "query_attention_time",
        "query": "query_attention_query",
    }.items():
        value = getattr(args, attr, None)
        if value is not None:
            query_cfg[key] = value
    if args.query_attention_max_frames is not None:
        query_cfg["max_frames"] = args.query_attention_max_frames
    if args.query_attention_fps is not None:
        query_cfg["fps"] = args.query_attention_fps
    if args.query_attention_alpha is not None:
        query_cfg["alpha"] = args.query_attention_alpha
    return query_cfg


def query_attention_output_path(args: argparse.Namespace) -> Path:
    if args.query_attention_output_dir:
        root = Path(args.query_attention_output_dir)
    else:
        root = Path(args.checkpoint).resolve().parent / "query_attention"
    return root / "query_attention.mp4"


def evaluate(args: argparse.Namespace) -> dict[str, float | int | str | list[str]]:
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    cfg, config_source = load_config_for_checkpoint(args.checkpoint, args.config)
    if args.dataset_root:
        cfg["dataset"].pop("roots", None)
        cfg["dataset"]["root"] = args.dataset_root
    cfg.setdefault("policy", {})["disparity_ablation"] = args.disparity_ablation
    if args.suppress_dynamo_errors:
        dynamo = importlib.import_module("torch._dynamo")
        dynamo.config.suppress_errors = True

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    loader = make_loader(cfg, args.batch_size, args.num_workers)
    model = load_model(cfg, args.checkpoint, device)
    model.eval()
    query_attention_cfg = resolve_query_attention_config(args, cfg)
    query_attention_frames = []

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
    raw_mse_sum = 0.0
    raw_mae_sum = 0.0
    dataset = loader.dataset
    action_mode = getattr(dataset, "action_mode", None)
    is_robotwin = action_mode == "relative-eef"

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
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
                if query_attention_cfg["enabled"] and len(query_attention_frames) < int(query_attention_cfg["max_frames"]):
                    pred, attention = model(left, right, state, return_attention=True)
                else:
                    pred = model(left, right, state)
                    attention = None
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

            if attention is not None:
                remaining = int(query_attention_cfg["max_frames"]) - len(query_attention_frames)
                frames = render_query_attention_frames(left, attention, query_attention_cfg)
                query_attention_frames.extend(frames[:remaining])

            if args.log_interval and (batch_idx + 1) % args.log_interval == 0:
                running_normalized_mse = normalized_mse_sum / max(total_elements, 1)
                if is_robotwin:
                    running_relative_mse = relative_mse_sum / max(total_elements, 1)
                    running_absolute_mse = absolute_mse_sum / max(total_elements, 1)
                    print(
                        f"batch={batch_idx + 1} samples={total_samples} "
                        f"normalized_mse={running_normalized_mse:.8f} "
                        f"relative_mse={running_relative_mse:.8f} "
                        f"absolute_mse={running_absolute_mse:.8f}",
                        flush=True,
                    )
                else:
                    running_raw_mse = raw_mse_sum / max(total_elements, 1)
                    print(
                        f"batch={batch_idx + 1} samples={total_samples} "
                        f"normalized_mse={running_normalized_mse:.8f} "
                        f"raw_mse={running_raw_mse:.8f}",
                        flush=True,
                    )

    if total_elements == 0:
        raise RuntimeError("No evaluation batches were processed.")

    normalized_mse = normalized_mse_sum / total_elements
    dataset_roots = cfg["dataset"].get("roots") or [cfg["dataset"]["root"]]
    metrics = {
        "checkpoint": str(args.checkpoint),
        "config_source": config_source,
        "disparity_ablation": args.disparity_ablation,
        "dataset_root": ", ".join(str(root) for root in dataset_roots),
        "dataset_roots": [str(root) for root in dataset_roots],
        "samples": total_samples,
        "action_elements": total_elements,
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
    if query_attention_cfg["enabled"] and query_attention_frames:
        output_path = query_attention_output_path(args)
        save_query_attention_video(
            query_attention_frames,
            output_path,
            fps=int(query_attention_cfg["fps"]),
        )
        metrics["query_attention_video"] = str(output_path)
        metrics["query_attention_frames"] = len(query_attention_frames)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default="outputs/rdt_v1_mixed/latest.pt")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-init", choices=["randn", "zeros"], default=None)
    parser.add_argument("--disparity-ablation", choices=["none", "zero", "shuffle"], default="none")
    add_boolean_optional_argument(parser, "--amp", default=True)
    add_boolean_optional_argument(parser, "--suppress-dynamo-errors", default=True)
    parser.add_argument("--visualize-query-attention", action="store_true")
    parser.add_argument("--query-attention-output-dir", default=None)
    parser.add_argument("--query-attention-mode", choices=["all", "single"], default=None)
    parser.add_argument("--query-attention-sources", default=None)
    parser.add_argument("--query-attention-view", default=None)
    parser.add_argument("--query-attention-time", default=None)
    parser.add_argument("--query-attention-query", default=None)
    parser.add_argument("--query-attention-max-frames", type=int, default=None)
    parser.add_argument("--query-attention-fps", type=int, default=None)
    parser.add_argument("--query-attention-alpha", type=float, default=None)
    args = parser.parse_args()

    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
