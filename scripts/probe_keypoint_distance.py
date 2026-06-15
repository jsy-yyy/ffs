from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ffs import load_config_for_checkpoint
from ffs.datasets import build_stereo_lerobot_dataset
from ffs.policies.builder import build_policy


def load_model(cfg: dict[str, Any], checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_policy(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def find_first_episode(cfg: dict[str, Any]) -> int:
    probe_cfg = dict(cfg["dataset"])
    probe_cfg.pop("episode_indices", None)
    probe_cfg["num_episodes"] = 1
    dataset = build_stereo_lerobot_dataset(probe_cfg, cfg["policy"], cfg.get("head", {}))
    if not getattr(dataset, "rows", None):
        raise RuntimeError("Dataset has no rows.")
    return int(dataset.rows[0]["episode_index"])


def get_pool_keypoints(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    pools = getattr(getattr(model, "adapter", None), "pools", {})
    for name, pool in pools.items():
        kps = getattr(pool, "kps", None)
        if kps is not None:
            out[name] = kps.detach().float().cpu()
    return out


def update_pair_stats(
    stats: dict[str, dict[str, float]],
    name: str,
    rgb_kps: torch.Tensor,
    disp_kps: torch.Tensor,
) -> None:
    if rgb_kps.shape != disp_kps.shape:
        raise ValueError(f"{name}: keypoint shapes differ: rgb={tuple(rgb_kps.shape)} disp={tuple(disp_kps.shape)}")

    aligned = torch.linalg.vector_norm(rgb_kps - disp_kps, dim=-1)
    nearest = torch.cdist(rgb_kps, disp_kps).min(dim=-1).values
    rgb_centroid = rgb_kps.mean(dim=1)
    disp_centroid = disp_kps.mean(dim=1)
    centroid = torch.linalg.vector_norm(rgb_centroid - disp_centroid, dim=-1)

    entry = stats[name]
    count = int(aligned.numel())
    sample_count = int(rgb_kps.shape[0])
    entry["count"] += count
    entry["samples"] += sample_count
    entry["aligned_sum"] += float(aligned.sum())
    entry["aligned_sq_sum"] += float((aligned * aligned).sum())
    entry["nearest_sum"] += float(nearest.sum())
    entry["nearest_sq_sum"] += float((nearest * nearest).sum())
    entry["centroid_sum"] += float(centroid.sum())
    entry["centroid_sq_sum"] += float((centroid * centroid).sum())
    entry["aligned_max"] = max(entry["aligned_max"], float(aligned.max()))
    entry["nearest_max"] = max(entry["nearest_max"], float(nearest.max()))
    entry["centroid_max"] = max(entry["centroid_max"], float(centroid.max()))


def finish_entry(entry: dict[str, float]) -> dict[str, float | int]:
    count = max(int(entry["count"]), 1)
    samples = max(int(entry["samples"]), 1)
    aligned_mean = entry["aligned_sum"] / count
    nearest_mean = entry["nearest_sum"] / count
    centroid_mean = entry["centroid_sum"] / samples
    return {
        "samples": int(entry["samples"]),
        "keypoints": int(entry["count"]),
        "aligned_l2_mean": aligned_mean,
        "aligned_l2_rmse": (entry["aligned_sq_sum"] / count) ** 0.5,
        "aligned_l2_max": entry["aligned_max"],
        "nearest_l2_mean": nearest_mean,
        "nearest_l2_rmse": (entry["nearest_sq_sum"] / count) ** 0.5,
        "nearest_l2_max": entry["nearest_max"],
        "centroid_l2_mean": centroid_mean,
        "centroid_l2_rmse": (entry["centroid_sq_sum"] / samples) ** 0.5,
        "centroid_l2_max": entry["centroid_max"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RGB and disparity SpatialSoftmax keypoints for one episode.")
    parser.add_argument("--checkpoint", default="outputs/native_dp_with_waft_disparity_noise/latest.pt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    cfg, config_source = load_config_for_checkpoint(args.checkpoint, args.config)
    if args.episode_index is None:
        episode_index = find_first_episode(cfg)
    else:
        episode_index = int(args.episode_index)

    dataset_cfg = dict(cfg["dataset"])
    dataset_cfg.pop("num_episodes", None)
    dataset_cfg["episode_indices"] = [episode_index]
    dataset = build_stereo_lerobot_dataset(dataset_cfg, cfg["policy"], cfg.get("head", {}))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device(args.device)
    model = load_model(cfg, args.checkpoint, device)
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "samples": 0.0,
            "aligned_sum": 0.0,
            "aligned_sq_sum": 0.0,
            "nearest_sum": 0.0,
            "nearest_sq_sum": 0.0,
            "centroid_sum": 0.0,
            "centroid_sq_sum": 0.0,
            "aligned_max": 0.0,
            "nearest_max": 0.0,
            "centroid_max": 0.0,
        }
    )

    total_windows = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            _ = model.encode(left, right, state)
            kps = get_pool_keypoints(model)
            for view in ("agentview", "robot0_eye_in_hand"):
                rgb_name = f"{view}_image"
                disp_name = f"{view}_disp"
                if rgb_name in kps and disp_name in kps:
                    update_pair_stats(stats, view, kps[rgb_name], kps[disp_name])
            total_windows += int(left.shape[0])
            print(f"processed batch {batch_idx + 1}/{len(loader)} windows={total_windows}", flush=True)

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve(strict=False)),
        "config_source": config_source,
        "episode_index": episode_index,
        "windows": total_windows,
        "observation_horizon": int(cfg["policy"].get("observation_horizon", cfg["policy"]["num_history_frames"])),
        "coordinate_range": "[-1, 1]",
        "note": "aligned compares same keypoint indices; nearest ignores keypoint ordering.",
        "pairs": {name: finish_entry(entry) for name, entry in stats.items()},
    }
    print(json.dumps(result, indent=2), flush=True)

    output = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent / "keypoint_distance_episode.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
