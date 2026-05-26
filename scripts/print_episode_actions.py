from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ffs import load_config
from ffs.datasets.lerobot import absolute_action_to_relative_eef_pose


def load_episode_rows(root: Path, episode: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = ["observation.state", "action", "episode_index", "frame_index"]
    for data_file in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(data_file, columns=columns).to_pydict()
        for i, episode_index in enumerate(table["episode_index"]):
            if int(episode_index) != episode:
                continue
            rows.append(
                {
                    "frame_index": int(table["frame_index"][i]),
                    "state": table["observation.state"][i],
                    "action": table["action"][i],
                    "data_file": data_file.relative_to(root).as_posix(),
                }
            )
    rows.sort(key=lambda row: row["frame_index"])
    if not rows:
        raise ValueError(f"No rows found for episode {episode} under {root}")
    return rows


def action_values(action: torch.Tensor, precision: int | None) -> list[float]:
    values = [float(value) for value in action.tolist()]
    if precision is None or precision < 0:
        return values
    return [round(value, precision) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print GT EEF actions for a LeRobot episode.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--action-space", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--relative", action="store_true", help="Print action relative to the previous state.")
    parser.add_argument("--jsonl", action="store_true", help="Print machine-readable JSONL records.")
    parser.add_argument("--output", default=None, help="Write records to a JSON or JSONL file instead of stdout.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(args.dataset_root or cfg["dataset"]["root"])
    rows = load_episode_rows(root, args.episode)
    end = len(rows) if args.limit <= 0 else min(len(rows), args.start + args.limit)

    mode = "relative_to_previous_state" if args.relative else "absolute"
    action_space = args.action_space or ("relative_ee_quat16" if args.relative else "absolute_ee_quat16")
    episode_id = args.episode if args.episode_id is None else args.episode_id
    records = []
    previous_action = None
    for row_idx in range(args.start, end):
        row = rows[row_idx]
        action = torch.tensor(row["action"], dtype=torch.float32)
        if args.relative:
            state = torch.tensor(row["state"], dtype=torch.float32)
            action = absolute_action_to_relative_eef_pose(action.unsqueeze(0), state)[0]

        delta_norm = None
        if previous_action is not None:
            delta_norm = float(torch.linalg.vector_norm(action - previous_action))
        previous_action = action

        values = action_values(action, args.precision)
        record = {
            "task_name": args.task_name,
            "seed": args.seed,
            "episode_id": episode_id,
            "prompt": args.prompt,
            "env_step": row["frame_index"],
            "chunk_index": row_idx,
            "timestep": row_idx,
            "action_space": action_space,
            "action": values,
            "left": values[:8],
            "right": values[8:],
            "action_norm": float(torch.linalg.vector_norm(action)),
            "delta_from_prev_timestep_norm": delta_norm,
        }
        records.append(record)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl = args.jsonl or output_path.suffix == ".jsonl"
        if write_jsonl:
            with output_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            payload = {
                "episode": args.episode,
                "root": str(root),
                "rows": len(rows),
                "start": args.start,
                "end": end,
                "mode": mode,
                "records": records,
            }
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"wrote {len(records)} records to {output_path}")
        return

    print(f"episode={args.episode} root={root} rows={len(rows)} showing=[{args.start}, {end}) mode={mode}")
    for record in records:
        if args.jsonl:
            print(json.dumps(record, ensure_ascii=False))
        else:
            print(
                f"timestep={record['timestep']:04d} env_step={record['env_step']:04d} "
                f"left={record['left']} right={record['right']}"
            )


if __name__ == "__main__":
    main()
