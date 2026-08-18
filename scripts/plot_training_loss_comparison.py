from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_RUNS = (
    ("topk2d_no_text", "train_robotwin_topk2d_tmux.log"),
    ("topk2d_task_id_text", "train_robotwin_topk2d_taskid_tmux.log"),
)

LOSS_RE = re.compile(r"global_step=(\d+)\s+loss=([0-9.eE+-]+)")
ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use LABEL=LOG_PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("--run must use non-empty LABEL=LOG_PATH")
    return label, Path(path)


def read_loss_points(path: Path) -> list[tuple[int, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    by_step: dict[int, float] = {}
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = LOSS_RE.search(line)
        if match:
            by_step[int(match.group(1))] = float(match.group(2))
    return sorted(by_step.items())


def time_weighted_ema(points: list[tuple[int, float]], alpha: float) -> list[float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("--ema-alpha must be in (0, 1).")
    if not points:
        return []

    smoothed = [points[0][1]]
    previous_step = points[0][0]
    previous_ema = points[0][1]
    for step, loss in points[1:]:
        step_delta = max(1, step - previous_step)
        decay = alpha ** step_delta
        previous_ema = decay * previous_ema + (1.0 - decay) * loss
        smoothed.append(previous_ema)
        previous_step = step
    return smoothed


def filter_steps(
    points: list[tuple[int, float]],
    *,
    min_step: int | None,
    max_step: int | None,
) -> list[tuple[int, float]]:
    filtered = points
    if min_step is not None:
        filtered = [(step, loss) for step, loss in filtered if step >= min_step]
    if max_step is not None:
        filtered = [(step, loss) for step, loss in filtered if step <= max_step]
    return filtered


def write_csv(path: Path, series: dict[str, list[tuple[int, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "global_step", "loss"])
        for label, points in series.items():
            for step, loss in points:
                writer.writerow([label, step, loss])


def plot_loss(
    output: Path,
    series: dict[str, list[tuple[int, float]]],
    *,
    title: str,
    yscale: str,
    ema_alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (label, points) in enumerate(series.items()):
        xs = [step for step, _ in points]
        ys = [loss for _, loss in points]
        color = colors[idx % len(colors)] if colors else None
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=3.2,
            linewidth=1.6,
            color=color,
            alpha=0.9,
            label=label,
        )
        if len(points) > 1:
            ax.plot(
                xs,
                time_weighted_ema(points, ema_alpha),
                linewidth=3.0,
                color=color,
                alpha=0.28,
            )

    ax.set_title(title, pad=12)
    ax.set_xlabel("global step")
    ax.set_ylabel("training loss")
    ax.set_yscale(yscale)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)
    ax.text(
        0.99,
        0.02,
        f"points are log_interval outputs; thick line = time-weighted EMA, alpha={ema_alpha}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot training loss curves from FFS train logs.",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        metavar="LABEL=LOG_PATH",
        help="Run label and log path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("loss_text_vs_notext_topk2d.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("loss_text_vs_notext_topk2d.csv"),
        help="Output CSV path for parsed points.",
    )
    parser.add_argument(
        "--title",
        default="WAFT-RDT top-k2d training loss: task_id text token vs no text",
        help="Plot title.",
    )
    parser.add_argument(
        "--yscale",
        choices=("log", "linear"),
        default="log",
        help="Y-axis scale.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.99,
        help="Per-step alpha for the time-weighted EMA smoothing line.",
    )
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = args.run or [(label, Path(path)) for label, path in DEFAULT_RUNS]

    series: dict[str, list[tuple[int, float]]] = {}
    for label, path in runs:
        points = filter_steps(
            read_loss_points(path),
            min_step=args.min_step,
            max_step=args.max_step,
        )
        if not points:
            raise ValueError(f"No loss points found for {label!r} in {path}")
        series[label] = points
        print(
            f"{label}: points={len(points)} "
            f"first={points[0][0]}:{points[0][1]:.6g} "
            f"last={points[-1][0]}:{points[-1][1]:.6g}"
        )

    write_csv(args.csv, series)
    plot_loss(
        args.output,
        series,
        title=args.title,
        yscale=args.yscale,
        ema_alpha=args.ema_alpha,
    )
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {args.csv.resolve()}")


if __name__ == "__main__":
    main()
