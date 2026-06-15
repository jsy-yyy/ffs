from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
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


def hist2d(points: np.ndarray, bins: int) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        points[:, 1],
        points[:, 0],
        bins=bins,
        range=[[-1.0, 1.0], [-1.0, 1.0]],
    )
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.reshape(-1)
    q = q.reshape(-1)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def summarize(points: np.ndarray) -> dict[str, Any]:
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    cov = np.cov(points.T)
    return {
        "count": int(points.shape[0]),
        "mean_xy": [float(mean[0]), float(mean[1])],
        "std_xy": [float(std[0]), float(std[1])],
        "cov_xy": [[float(v) for v in row] for row in cov],
    }


def plot_distributions(
    distributions: dict[str, dict[str, np.ndarray]],
    output: Path,
    bins: int,
) -> None:
    views = list(distributions)
    fig, axes = plt.subplots(len(views), 3, figsize=(10.5, 3.4 * len(views)), dpi=180, squeeze=False)
    extent = [-1, 1, -1, 1]

    for row, view in enumerate(views):
        rgb_hist = hist2d(distributions[view]["rgb"], bins)
        disp_hist = hist2d(distributions[view]["disp"], bins)
        vmax = max(float(rgb_hist.max()), float(disp_hist.max()), 1e-9)
        diff = disp_hist - rgb_hist
        diff_abs = max(float(np.abs(diff).max()), 1e-9)

        for col, (title, image, cmap, vmin, vmax_i) in enumerate(
            [
                (f"{view} RGB", rgb_hist, "viridis", 0.0, vmax),
                (f"{view} disparity", disp_hist, "viridis", 0.0, vmax),
                (f"{view} disp - RGB", diff, "coolwarm", -diff_abs, diff_abs),
            ]
        ):
            ax = axes[row][col]
            im = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax_i,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_title(title, fontsize=9)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RGB vs disparity SpatialSoftmax keypoint distributions.")
    parser.add_argument("--checkpoint", default="outputs/native_dp_with_waft_disparity_noise/latest.pt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg, config_source = load_config_for_checkpoint(args.checkpoint, args.config)
    episode_index = find_first_episode(cfg) if args.episode_index is None else int(args.episode_index)

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
    collected: dict[str, dict[str, list[torch.Tensor]]] = {
        "agentview": {"rgb": [], "disp": []},
        "robot0_eye_in_hand": {"rgb": [], "disp": []},
    }

    windows = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            _ = model.encode(left, right, state)
            kps = get_pool_keypoints(model)
            for view in collected:
                rgb_name = f"{view}_image"
                disp_name = f"{view}_disp"
                if rgb_name in kps and disp_name in kps:
                    collected[view]["rgb"].append(kps[rgb_name].reshape(-1, 2))
                    collected[view]["disp"].append(kps[disp_name].reshape(-1, 2))
            windows += int(left.shape[0])
            print(f"processed batch {batch_idx + 1}/{len(loader)} windows={windows}", flush=True)

    distributions: dict[str, dict[str, np.ndarray]] = {}
    for view, modalities in collected.items():
        if modalities["rgb"] and modalities["disp"]:
            distributions[view] = {
                "rgb": torch.cat(modalities["rgb"], dim=0).numpy(),
                "disp": torch.cat(modalities["disp"], dim=0).numpy(),
            }
    if not distributions:
        raise RuntimeError("No RGB/disparity keypoint pairs were collected.")

    output = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent / "keypoint_distribution_episode.png"
    plot_distributions(distributions, output, args.bins)

    summary: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve(strict=False)),
        "config_source": config_source,
        "episode_index": episode_index,
        "windows": windows,
        "coordinate_range": "[-1, 1]",
        "bins": args.bins,
        "views": {},
    }
    for view, modalities in distributions.items():
        rgb_hist = hist2d(modalities["rgb"], args.bins)
        disp_hist = hist2d(modalities["disp"], args.bins)
        rgb_mean = modalities["rgb"].mean(axis=0)
        disp_mean = modalities["disp"].mean(axis=0)
        summary["views"][view] = {
            "rgb": summarize(modalities["rgb"]),
            "disp": summarize(modalities["disp"]),
            "mean_shift_l2": float(np.linalg.norm(rgb_mean - disp_mean)),
            "hist_l1": float(np.abs(rgb_hist - disp_hist).sum()),
            "hist_jsd_bits": js_divergence(rgb_hist, disp_hist),
        }

    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
