from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> dict[str, list[float]]:
    metrics = {"global_step": [], "epoch": [], "loss": [], "lr": []}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {line}") from exc

            metrics["global_step"].append(int(record["global_step"]))
            metrics["epoch"].append(int(record["epoch"]))
            metrics["loss"].append(float(record["loss"]))
            metrics["lr"].append(float(record["lr"]))
    if not metrics["global_step"]:
        raise ValueError(f"No metrics found in {path}")
    return metrics


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
            out.append(total / window)
        else:
            out.append(total / (i + 1))
    return out


def plot_metrics(metrics_path: Path, output_path: Path, smooth: int) -> None:
    metrics = load_metrics(metrics_path)
    steps = metrics["global_step"]
    loss = metrics["loss"]
    smoothed_loss = moving_average(loss, smooth)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.plot(steps, loss, color="#9ca3af", linewidth=0.8, alpha=0.55, label="loss")
    if smooth > 1:
        ax.plot(steps, smoothed_loss, color="#2563eb", linewidth=1.8, label=f"loss, ma{smooth}")

    ax.set_title(metrics_path.as_posix())
    ax.set_xlabel("global_step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/default/metrics.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth", type=int, default=20)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    output_path = Path(args.output) if args.output else metrics_path.with_suffix(".png")
    plot_metrics(metrics_path, output_path, args.smooth)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
