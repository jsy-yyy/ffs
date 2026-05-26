from __future__ import annotations

import argparse
import copy
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ffs import load_config
from ffs.datasets import LeRobotStereoDataset
from ffs.policies.stereo_action_policy import build_policy


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_dist() -> tuple[int, int, int, torch.device]:
    if not is_dist():
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        return 0, 0, 1, device

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def cleanup_dist() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        try:
            dist.destroy_process_group()
        except Exception as exc:
            print(f"warning: rank {rank} failed to destroy process group cleanly: {exc}")


def make_grad_scaler(use_amp: bool) -> Any:
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(device: torch.device, use_amp: bool) -> Any:
    return torch.amp.autocast(device_type=device.type, enabled=use_amp)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    step: int,
    next_step: int,
    global_step: int,
    cfg: dict[str, Any],
) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "next_step": next_step,
            "global_step": global_step,
            "config": cfg,
        },
        path,
    )
    if path.name == "latest.pt":
        eval_cfg = copy.deepcopy(cfg)
        if isinstance(eval_cfg.get("train"), dict):
            eval_cfg["train"]["resume_from"] = None
        with path.with_suffix(".yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(eval_cfg, f, sort_keys=False, allow_unicode=True)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    is_main: bool = True,
) -> tuple[int, int, int]:
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    next_step = int(ckpt.get("next_step", int(ckpt.get("step", -1)) + 1))
    global_step = int(ckpt.get("global_step", 0))
    if is_main:
        print(f"resumed from {path}: epoch={epoch} step={next_step} global_step={global_step}")
    return epoch, next_step, global_step


def prepare_metrics_file(path: Path, resume_from: str | Path | None, global_step: int) -> None:
    if not resume_from:
        path.write_text("", encoding="utf-8")
        return
    if not path.exists():
        return

    kept_lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(record.get("global_step", -1)) <= global_step:
                kept_lines.append(line)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(kept_lines)


def make_loader(cfg: dict[str, Any], rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
    policy_cfg = cfg["policy"]
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
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
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader, sampler


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    if bool(train_cfg.get("suppress_dynamo_errors", False)):
        dynamo = importlib.import_module("torch._dynamo")
        dynamo.config.suppress_errors = True

    rank, local_rank, world_size, device = setup_dist()
    is_main = rank == 0

    output_dir = Path(train_cfg["output_dir"])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    loader, sampler = make_loader(cfg, rank, world_size)
    model = build_policy(cfg).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(use_amp)

    start_epoch = 0
    start_step = 0
    global_step = 0
    resume_from = args.resume_from or train_cfg.get("resume_from")
    if resume_from:
        start_epoch, start_step, global_step = load_checkpoint(
            resume_from, model, optimizer, scaler, device, is_main=is_main
        )
        if start_step >= len(loader):
            start_epoch += 1
            start_step = 0
    elif is_main:
        print("starting fresh: no resume checkpoint configured")

    metrics_path = output_dir / "metrics.jsonl"
    if is_main:
        prepare_metrics_file(metrics_path, resume_from, global_step)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    stop_step = global_step + args.max_steps if args.max_steps is not None else None
    log_interval = int(train_cfg["log_interval"])
    save_ckpt_interval = train_cfg.get("save_ckpt_interval")
    save_ckpt_interval = int(save_ckpt_interval) if save_ckpt_interval else 0
    done = False
    last_epoch = start_epoch
    last_step = start_step - 1

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        last_epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        skip_steps = start_step if epoch == start_epoch else 0

        for step, batch in enumerate(loader):
            if step < skip_steps:
                continue
            last_step = step

            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                loss = raw_model.training_loss(left, right, state, action)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            loss_value = float(loss.detach().cpu())
            lr = optimizer.param_groups[0]["lr"]

            if is_main and global_step % log_interval == 0:
                print(f"epoch={epoch} step={step} global_step={global_step} loss={loss_value:.6f} lr={lr:.6g}")
            if is_main:
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": step,
                                "global_step": global_step,
                                "loss": loss_value,
                                "lr": lr,
                            }
                        )
                        + "\n"
                    )

            if is_main and save_ckpt_interval > 0 and global_step % save_ckpt_interval == 0:
                save_checkpoint(output_dir / "latest.pt", model, optimizer, scaler,
                                epoch, step, step + 1, global_step, cfg)
                save_checkpoint(output_dir / f"step_{global_step}.pt", model, optimizer, scaler,
                                epoch, step, step + 1, global_step, cfg)

            if stop_step is not None and global_step >= stop_step:
                done = True
                break

        if done:
            break

    if is_main and global_step > 0:
        final_epoch = last_epoch if done else last_epoch + 1
        final_step = last_step if done else -1
        final_next_step = last_step + 1 if done else 0
        save_checkpoint(output_dir / "latest.pt", model, optimizer, scaler,
                        final_epoch, final_step, final_next_step, global_step, cfg)

    del model, optimizer, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cleanup_dist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
