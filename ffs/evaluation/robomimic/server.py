from __future__ import annotations

import argparse
import socket
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import EvalConfig, load_eval_config
from .env_utils import payload_image_name
from .wire import recv_msg, send_msg


class RobomimicFFSService:
    def __init__(
        self,
        ffs_root: str | Path,
        checkpoint: str | Path,
        config_path: str | Path | None,
        device: str,
        amp: bool = True,
        sample_init: str | None = None,
        disparity_ablation: str = "none",
        debug_actions: bool = False,
    ) -> None:
        self.ffs_root = Path(ffs_root)
        if str(self.ffs_root) not in sys.path:
            sys.path.insert(0, str(self.ffs_root))

        from ffs import load_config_for_checkpoint
        from ffs.datasets import ActionNormalizer
        from ffs.policies.stereo_action_policy import build_policy

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for robomimic eval server, but it is unavailable.")

        self.cfg, self.config_source = load_config_for_checkpoint(checkpoint, config_path)
        self.cfg.setdefault("policy", {})["disparity_ablation"] = disparity_ablation
        self.camera_pairs = self.cfg["dataset"]["camera_pairs"]
        self.image_size = tuple(self.cfg["dataset"].get("image_size", ())) or None
        self.num_history_frames = int(self.cfg["policy"]["num_history_frames"])
        self.action_horizon = int(self.cfg["policy"]["action_horizon"])
        self.action_dim = int(self.cfg["policy"]["action_dim"])
        self.state_dim = int(self.cfg["policy"]["state_dim"])
        self.amp = bool(amp)
        self.debug_actions = bool(debug_actions)
        self.history: list[dict[str, Any]] = []
        self.session = {
            "task_name": "unknown",
            "seed": "unknown",
            "episode_id": "unknown",
        }

        self.action_normalizer = ActionNormalizer(
            self.cfg["dataset"].get("action_normalization"),
            self.action_dim,
        )
        self.model = build_policy(self.cfg).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()
        if sample_init is not None and hasattr(self.model.action_head, "sample_init"):
            self.model.action_head.sample_init = sample_init

    def reset(
        self,
        task_name: str | None = None,
        seed: int | str | None = None,
        episode_id: int | str | None = None,
    ) -> None:
        self.history = []
        self.session = {
            "task_name": task_name or "unknown",
            "seed": "unknown" if seed is None else seed,
            "episode_id": "unknown" if episode_id is None else episode_id,
        }
        self.model.eval()

    def predict(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._append_obs(obs)
        obs_window = self._history_window()
        left, right, state = self._make_batch(obs_window)
        use_amp = self.amp and self.device.type == "cuda"
        autocast = torch.amp.autocast(device_type=self.device.type, enabled=use_amp) if hasattr(torch, "amp") else nullcontext()
        with torch.inference_mode(), autocast:
            normalized = self.model(left, right, state)
        actions = self.action_normalizer.denormalize_action(normalized[0].float()).detach().cpu().numpy()
        if actions.shape != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"Expected action chunk shape {(self.action_horizon, self.action_dim)}, got {actions.shape}."
            )
        if self.debug_actions:
            print(
                "[robomimic FFS] "
                f"task={self.session['task_name']} seed={self.session['seed']} "
                f"episode={self.session['episode_id']} step={obs.get('step')} "
                f"actions_shape={actions.shape} first={np.array2string(actions[0], precision=4)}",
                flush=True,
            )
        return {
            "actions": actions.astype(np.float32, copy=False),
            "action_space": "robomimic_delta7",
            "metadata": {
                "config_source": self.config_source,
                "history": len(obs_window),
            },
        }

    def update(self, observations: list[dict[str, Any]]) -> None:
        for item in observations:
            obs = item.get("obs") if isinstance(item, dict) and "obs" in item else item
            self._append_obs(obs)

    def _append_obs(self, obs: dict[str, Any]) -> None:
        if "state" not in obs or "images" not in obs:
            raise KeyError("Observation payload must contain 'state' and 'images'.")
        step = int(obs.get("step", len(self.history)))
        obs["step"] = step
        if self.history and int(self.history[-1].get("step", -1)) == step:
            self.history[-1] = obs
        else:
            self.history.append(obs)
        if len(self.history) > self.num_history_frames:
            self.history = self.history[-self.num_history_frames :]

    def _history_window(self) -> list[dict[str, Any]]:
        if not self.history:
            raise RuntimeError("Cannot predict before receiving at least one observation.")
        window = list(self.history[-self.num_history_frames :])
        while len(window) < self.num_history_frames:
            window.insert(0, window[0])
        return window

    def _make_batch(self, obs_window: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left_views = []
        right_views = []
        for left_key, right_key in self.camera_pairs:
            left_name = payload_image_name(left_key)
            right_name = payload_image_name(right_key)
            left_views.append(torch.stack([self._image_tensor(obs, left_name) for obs in obs_window]))
            right_views.append(torch.stack([self._image_tensor(obs, right_name) for obs in obs_window]))

        state = torch.from_numpy(
            np.stack([np.asarray(obs["state"], dtype=np.float32) for obs in obs_window], axis=0)
        )
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {state.shape[-1]}.")
        left = torch.stack(left_views, dim=1).unsqueeze(0).to(self.device)
        right = torch.stack(right_views, dim=1).unsqueeze(0).to(self.device)
        state = state.unsqueeze(0).to(self.device)
        return left, right, state

    def _image_tensor(self, obs: dict[str, Any], image_name: str) -> torch.Tensor:
        try:
            image = np.asarray(obs["images"][image_name])
        except KeyError as exc:
            available = sorted(obs.get("images", {}).keys())
            raise KeyError(f"Missing image {image_name!r}; available images: {available}") from exc
        if image.ndim != 3:
            raise ValueError(f"Expected image {image_name!r} to have 3 dims, got {image.shape}.")
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            tensor = torch.from_numpy(image.copy()).float()
        else:
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            if image.shape[-1] != 3:
                raise ValueError(f"Expected RGB image {image_name!r}, got shape {image.shape}.")
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
        if self.image_size is not None:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor


def _config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {"policy": {}}
    for arg_name, cfg_name in {
        "host": "host",
        "port": "port",
        "checkpoint": "checkpoint",
        "config_path": "config_path",
        "device": "device",
        "sample_init": "sample_init",
        "disparity_ablation": "disparity_ablation",
    }.items():
        value = getattr(args, arg_name)
        if value is not None:
            overrides["policy"][cfg_name] = value
    if args.no_amp:
        overrides["policy"]["amp"] = False
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FFS robomimic policy server")
    parser.add_argument("--config", default="configs/robomimic_square_eval.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-init", choices=["randn", "zeros"], default=None)
    parser.add_argument("--disparity-ablation", choices=["none", "zero", "shuffle"], default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def build_service(config: EvalConfig) -> RobomimicFFSService:
    return RobomimicFFSService(
        ffs_root=config.policy.ffs_root,
        checkpoint=config.policy.checkpoint,
        config_path=config.policy.config_path,
        device=config.policy.device,
        amp=config.policy.amp,
        sample_init=config.policy.sample_init,
        disparity_ablation=config.policy.disparity_ablation,
        debug_actions=config.policy.debug_actions,
    )


def main() -> None:
    args = parse_args()
    config = load_eval_config(args.config, _config_overrides(args))
    service = build_service(config)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((config.policy.host, config.policy.port))
        server.listen(1)
        print(
            "robomimic FFS server listening on "
            f"{config.policy.host}:{config.policy.port} "
            f"checkpoint={config.policy.checkpoint} device={config.policy.device} "
            f"config={service.config_source}",
            flush=True,
        )
        while True:
            conn, addr = server.accept()
            print(f"robomimic client connected: {addr}", flush=True)
            with conn:
                while True:
                    try:
                        req = recv_msg(conn)
                    except ConnectionError:
                        break
                    try:
                        cmd = req["cmd"]
                        if cmd == "reset":
                            service.reset(
                                task_name=req.get("task_name"),
                                seed=req.get("seed"),
                                episode_id=req.get("episode_id"),
                            )
                            send_msg(conn, {"ok": True})
                        elif cmd == "predict":
                            send_msg(conn, service.predict(req["obs"]))
                        elif cmd == "update":
                            service.update(req.get("observations", []))
                            send_msg(conn, {"ok": True})
                        elif cmd == "ping":
                            send_msg(conn, {"ok": True})
                        else:
                            send_msg(conn, {"error": f"unknown cmd: {cmd}"})
                    except Exception:
                        send_msg(conn, {"error": traceback.format_exc()})
                        break
            print("robomimic client disconnected", flush=True)


if __name__ == "__main__":
    main()

