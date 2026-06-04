from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

PLOT_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#B58900",
    "#332288",
    "#AA4499",
    "#44AA99",
    "#117733",
    "#882255",
    "#88CCEE",
    "#DDCC77",
    "#999933",
]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
LINE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


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


def limit_metrics(metrics: dict[str, list[float]], max_steps: int | None, path: Path) -> dict[str, list[float]]:
    if max_steps is None:
        return metrics

    limited = {"global_step": [], "loss": []}
    for step, loss in zip(metrics["global_step"], metrics["loss"]):
        if step <= max_steps:
            limited["global_step"].append(step)
            limited["loss"].append(loss)

    if not limited["global_step"]:
        raise ValueError(f"No metrics in {path} have global_step <= {max_steps}")
    return limited


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


def marker_stride(num_points: int) -> int:
    if num_points <= 0:
        return 1
    return max(1, num_points // 14)


def spread_end_labels(
    endpoints: Sequence[tuple[float, float, str, str]],
    y_min: float,
    y_max: float,
    log_scale: bool = False,
) -> list[tuple[float, float, str, str, float]]:
    if not endpoints:
        return []

    def to_axis_y(value: float) -> float:
        if not log_scale:
            return value
        return math.log10(max(value, y_min, 1e-300))

    def from_axis_y(value: float) -> float:
        if not log_scale:
            return value
        return 10**value

    axis_min = to_axis_y(y_min)
    axis_max = to_axis_y(y_max)
    y_range = max(axis_max - axis_min, 1e-12)
    min_gap = y_range * 0.035
    sorted_endpoints = sorted(endpoints, key=lambda item: to_axis_y(item[1]))
    adjusted: list[tuple[float, float, str, str, float]] = []
    prev_y = axis_min - min_gap

    for x, y, label, color in sorted_endpoints:
        adjusted_y = min(max(to_axis_y(y), prev_y + min_gap), axis_max)
        adjusted.append((x, y, label, color, adjusted_y))
        prev_y = adjusted_y

    overflow = adjusted[-1][-1] - axis_max
    if overflow > 0:
        adjusted = [(x, y, label, color, y_label - overflow) for x, y, label, color, y_label in adjusted]
    return [(x, y, label, color, from_axis_y(y_label)) for x, y, label, color, y_label in adjusted]


def plot_metrics(
    metrics_paths: Sequence[Path],
    output_path: Path,
    smooth: int,
    labels: Sequence[str],
    max_steps: int | None,
    log_scale: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    runs = []

    for i, (metrics_path, label) in enumerate(zip(metrics_paths, labels)):
        metrics = limit_metrics(load_metrics(metrics_path), max_steps, metrics_path)
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        marker = LINE_MARKERS[i % len(LINE_MARKERS)]
        runs.append((metrics["global_step"], metrics["loss"], label, color, line_style, marker, i))

    if log_scale and not any(loss > 0 for _, losses, *_ in runs for loss in losses):
        raise ValueError("--log requires at least one positive loss value")

    if smooth > 1:
        raw_alpha = 0.08 if len(runs) > 1 else 0.22
        for steps, loss, *_ in runs:
            ax.plot(steps, loss, color="#6b7280", linewidth=0.18, alpha=raw_alpha, label="_nolegend_", zorder=1)

    endpoints = []
    plotted_runs = []
    line_effects = [
        path_effects.Stroke(linewidth=1.05, foreground="white", alpha=0.85),
        path_effects.Normal(),
    ]
    for steps, loss, label, color, line_style, marker, run_index in runs:
        y_values = moving_average(loss, smooth) if smooth > 1 else loss
        ax.plot(
            steps,
            y_values,
            color=color,
            linestyle=line_style,
            linewidth=0.55 if smooth > 1 else 0.45,
            alpha=0.95,
            marker=marker,
            markersize=2.6,
            markevery=(run_index % marker_stride(len(steps)), marker_stride(len(steps))),
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=label,
            path_effects=line_effects,
            zorder=3,
        )
        endpoints.append((float(steps[-1]), float(y_values[-1]), label, color))
        plotted_runs.append((steps, y_values, color, marker, run_index))

    for steps, y_values, color, marker, run_index in plotted_runs:
        stride = marker_stride(len(steps))
        ax.plot(
            steps,
            y_values,
            color=color,
            linestyle="None",
            marker=marker,
            markersize=2.8,
            markevery=(run_index % stride, stride),
            markerfacecolor="white",
            markeredgewidth=0.8,
            label="_nolegend_",
            zorder=4,
        )

    ax.set_title("Loss comparison")
    ax.set_xlabel("global_step")
    ax.set_ylabel("loss (log scale)" if log_scale else "loss")
    if log_scale:
        ax.set_yscale("log", nonpositive="clip")
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25)

    x_values = [step for steps, *_ in runs for step in steps]
    if x_values:
        x_min, x_max = min(x_values), max(x_values)
        x_range = max(x_max - x_min, 1)
        ax.set_xlim(x_min, x_max + x_range * (0.18 if len(runs) <= 12 else 0.08))

    if len(runs) <= 12:
        y_min, y_max = ax.get_ylim()
        for x, y, label, color, y_label in spread_end_labels(endpoints, y_min, y_max, log_scale):
            ax.annotate(
                label,
                xy=(x, max(y, y_min) if log_scale else y),
                xytext=(x + max((ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012, 1), y_label),
                textcoords="data",
                color=color,
                fontsize=8,
                va="center",
                ha="left",
                arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.45, "linewidth": 0.6},
                path_effects=[
                    path_effects.Stroke(linewidth=2.5, foreground="white", alpha=0.95),
                    path_effects.Normal(),
                ],
                clip_on=False,
            )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False, fontsize=8)
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
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log", action="store_true", help="plot loss on a logarithmic y axis")
    args = parser.parse_args()

    metrics_paths = [Path(path) for path in args.metrics]
    if args.labels is not None and len(args.labels) != len(metrics_paths):
        parser.error(f"--labels expects {len(metrics_paths)} value(s), got {len(args.labels)}")
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("--max-steps must be >= 0")
    labels = args.labels if args.labels is not None else [default_label(path) for path in metrics_paths]
    output_path = Path(args.output) if args.output else default_output_path(metrics_paths)
    plot_metrics(metrics_paths, output_path, args.smooth, labels, args.max_steps, args.log)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
