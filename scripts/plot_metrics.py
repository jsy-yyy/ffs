from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

PLOT_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#7c3aed",
    "#0f766e",
]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


def load_metrics(path: Path) -> dict[str, list[float]]:
    metrics = {"global_step": [], "loss": []}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_no}: {line}") from exc

            missing = [key for key in ("global_step", "loss") if key not in record]
            if missing:
                raise ValueError(f"{path}: missing {', '.join(missing)} on line {line_no}")
            try:
                metrics["global_step"].append(int(record["global_step"]))
                metrics["loss"].append(float(record["loss"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid global_step or loss on line {line_no}: {line}") from exc
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


def default_label(metrics_path: Path) -> str:
    if metrics_path.name == "metrics.jsonl" and metrics_path.parent.parent.name == "outputs":
        return metrics_path.parent.name
    return metrics_path.stem


def default_output_path(metrics_paths: Sequence[Path]) -> Path:
    if len(metrics_paths) == 1:
        return metrics_paths[0].with_suffix(".png")

    run_dirs = [path.parent for path in metrics_paths]
    try:
        common_parent = Path(os.path.commonpath(run_dirs))
    except ValueError:
        common_parent = Path(os.path.commonpath([path.resolve() for path in run_dirs]))
    return common_parent / "metrics_compare.png"


def plot_metrics(metrics_paths: Sequence[Path], output_path: Path, smooth: int, labels: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    runs = []

    for i, (metrics_path, label) in enumerate(zip(metrics_paths, labels)):
        metrics = load_metrics(metrics_path)
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        runs.append((metrics["global_step"], metrics["loss"], label, color, line_style))

    if smooth > 1:
        raw_alpha = 0.08 if len(runs) > 1 else 0.22
        for steps, loss, _, _, _ in runs:
            ax.plot(steps, loss, color="#6b7280", linewidth=0.35, alpha=raw_alpha, label="_nolegend_", zorder=1)

    for steps, loss, label, color, line_style in runs:
        y_values = moving_average(loss, smooth) if smooth > 1 else loss
        ax.plot(
            steps,
            y_values,
            color=color,
            linestyle=line_style,
            linewidth=1.2 if smooth > 1 else 0.9,
            alpha=0.95,
            label=label,
            zorder=3,
        )

    ax.set_title("Loss comparison")
    ax.set_xlabel("global_step")
    ax.set_ylabel("loss")
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", default=["outputs/default/metrics.jsonl"])
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth", type=int, default=20)
    args = parser.parse_args()

    metrics_paths = [Path(path) for path in args.metrics]
    if args.labels is not None and len(args.labels) != len(metrics_paths):
        parser.error(f"--labels expects {len(metrics_paths)} value(s), got {len(args.labels)}")
    labels = args.labels if args.labels is not None else [default_label(path) for path in metrics_paths]
    output_path = Path(args.output) if args.output else default_output_path(metrics_paths)
    plot_metrics(metrics_paths, output_path, args.smooth, labels)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
