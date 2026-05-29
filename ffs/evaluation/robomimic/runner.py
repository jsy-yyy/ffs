from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np

from .config import EvalConfig, load_ffs_sidecar_config
from .env_utils import (
    apply_runtime_env,
    camera_names_from_pairs,
    check_success,
    decode_attr,
    get_lowdim_obs,
    load_env_meta,
    make_observation_payload,
    make_robosuite_env,
    payload_image_name,
    reset_to,
    sorted_demo_names,
)
from .wire import recv_msg, send_msg


@dataclass
class EpisodeResult:
    task_name: str
    seed: int
    episode_id: int
    demo_name: str | None
    success: bool
    horizon: int
    ret: float


class RobomimicPolicyClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)
        self.sock: socket.socket | None = None

    def setup(self) -> None:
        self.sock = socket.create_connection((self.host, self.port))
        send_msg(self.sock, {"cmd": "ping"})
        self._recv_ok()

    def reset(self, task_name: str, seed: int, episode_id: int) -> None:
        send_msg(
            self._socket(),
            {
                "cmd": "reset",
                "task_name": task_name,
                "seed": int(seed),
                "episode_id": int(episode_id),
            },
        )
        self._recv_ok()

    def predict(self, obs: dict[str, Any]) -> tuple[np.ndarray, str, dict[str, Any]]:
        send_msg(self._socket(), {"cmd": "predict", "obs": obs})
        ret = self._recv_ok()
        return np.asarray(ret["actions"], dtype=np.float32), str(ret["action_space"]), dict(ret.get("metadata", {}))

    def update(self, observations: list[dict[str, Any]]) -> None:
        send_msg(self._socket(), {"cmd": "update", "observations": observations})
        self._recv_ok()

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _socket(self) -> socket.socket:
        if self.sock is None:
            raise RuntimeError("RobomimicPolicyClient.setup() must be called first.")
        return self.sock

    def _recv_ok(self) -> dict[str, Any]:
        ret = recv_msg(self._socket())
        if "error" in ret:
            raise RuntimeError(ret["error"])
        return ret


def select_action_steps(actions: np.ndarray, execute_chunk_steps: int | None) -> np.ndarray:
    action_steps = np.asarray(actions, dtype=np.float32).reshape(-1, actions.shape[-1])
    if execute_chunk_steps is None:
        return action_steps
    steps = int(execute_chunk_steps)
    if steps <= 0:
        raise ValueError("execute_chunk_steps must be positive or null.")
    return action_steps[:steps]


class RobomimicEvaluator:
    def __init__(self, config: EvalConfig, policy: RobomimicPolicyClient | None = None) -> None:
        self.config = config
        self.policy = policy or RobomimicPolicyClient(config.policy.host, config.policy.port)
        self.ffs_cfg = load_ffs_sidecar_config(config.policy.checkpoint, config.policy.config_path)
        self.camera_pairs = self.ffs_cfg["dataset"]["camera_pairs"]
        self.camera_names = camera_names_from_pairs(self.camera_pairs)
        self.state_dim = int(self.ffs_cfg["policy"]["state_dim"])
        self.action_dim = int(self.ffs_cfg["policy"]["action_dim"])
        self.image_size = tuple(self.ffs_cfg["dataset"].get("image_size", (224, 224)))
        self.camera_height = int(config.env.camera_height or self.image_size[0])
        self.camera_width = int(config.env.camera_width or self.image_size[1])
        self.stereo_baseline = float(config.env.stereo_baseline or 0.06)
        self.dataset_path = Path(config.env.dataset_path)
        self.env_meta = load_env_meta(self.dataset_path)
        self.task_name = str(self.env_meta.get("env_name", "robomimic"))
        self.demo_names = sorted_demo_names(self.dataset_path)

    def dry_run_summary(self) -> dict[str, Any]:
        return {
            "checkpoint": self.config.policy.checkpoint,
            "config_path": str(self.config.policy.config_path or "sidecar"),
            "dataset_path": str(self.dataset_path),
            "task_name": self.task_name,
            "camera_names": self.camera_names,
            "camera_pairs": self.camera_pairs,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_demos": len(self.demo_names),
            "first_demo": self.demo_names[0] if self.demo_names else None,
            "render": {
                "height": self.camera_height,
                "width": self.camera_width,
                "baseline": self.stereo_baseline,
                "mujoco_gl": self.config.env.mujoco_gl,
                "render_gpu_device_id": self.config.env.render_gpu_device_id,
            },
        }

    def run(self) -> dict[str, float]:
        apply_runtime_env(
            numba_disable_jit=self.config.env.numba_disable_jit,
            mujoco_gl=self.config.env.mujoco_gl,
        )
        env = make_robosuite_env(
            self.env_meta,
            camera_names=self.camera_names,
            camera_height=self.camera_height,
            camera_width=self.camera_width,
            render_gpu_device_id=self.config.env.render_gpu_device_id,
        )
        self.policy.setup()
        results: list[EpisodeResult] = []
        try:
            for episode_id in range(int(self.config.env.n_rollouts)):
                result = self.run_episode(env, episode_id)
                results.append(result)
                self._write_episode(result)
                print(
                    f"{self.task_name} episode={episode_id} "
                    f"success={result.success} horizon={result.horizon} return={result.ret:.4f}",
                    flush=True,
                )
            metrics = self._summarize(results)
            self._write_summary(metrics)
            return metrics
        finally:
            self.policy.close()
            if hasattr(env, "close"):
                env.close()

    def run_episode(self, env, episode_id: int) -> EpisodeResult:
        seed = int(self.config.env.seed) + int(episode_id)
        np.random.seed(seed)
        demo_name = None
        if self.config.env.reset_mode == "dataset":
            demo_name = self._reset_to_dataset_demo(env, episode_id)
            lowdim_obs = get_lowdim_obs(env)
        elif self.config.env.reset_mode == "random":
            if hasattr(env, "seed"):
                env.seed(seed)
            lowdim_obs = env.reset()
        else:
            raise ValueError("env.reset_mode must be one of: dataset, random.")

        self.policy.reset(self.task_name, seed=seed, episode_id=episode_id)
        step = 0
        ret = 0.0
        success = False
        done = False
        action_records: list[dict[str, Any]] = []
        video_frames: list[np.ndarray] = []
        obs_payload = self._payload(env, lowdim_obs, step)
        self._maybe_append_video_frame(video_frames, obs_payload, step)

        while step < int(self.config.env.horizon) and not done and not success:
            actions, action_space, _ = self.policy.predict(obs_payload)
            if action_space != "robomimic_delta7":
                raise ValueError(f"Unsupported action_space from server: {action_space}")
            updates = []
            for action in select_action_steps(actions, self.config.action.execute_chunk_steps):
                lowdim_obs, reward, done, _info = env.step(action)
                step += 1
                ret += float(reward)
                success = check_success(env)
                obs_payload = self._payload(env, lowdim_obs, step)
                updates.append(obs_payload)
                self._maybe_append_video_frame(video_frames, obs_payload, step)
                if self.config.record.save_action_jsonl:
                    action_records.append(
                        {
                            "episode_id": int(episode_id),
                            "seed": int(seed),
                            "step": int(step),
                            "action": [float(value) for value in action.tolist()],
                            "reward": float(reward),
                            "success": bool(success),
                            "done": bool(done),
                        }
                    )
                if step >= int(self.config.env.horizon) or done or success:
                    break
            if updates:
                self.policy.update(updates)

        if self.config.record.save_action_jsonl and action_records:
            self._write_actions(episode_id, seed, action_records)
        if self.config.record.save_video and video_frames:
            self._write_video(episode_id, seed, video_frames, bool(success))
        return EpisodeResult(
            task_name=self.task_name,
            seed=seed,
            episode_id=episode_id,
            demo_name=demo_name,
            success=bool(success),
            horizon=int(step),
            ret=float(ret),
        )

    def _reset_to_dataset_demo(self, env, episode_id: int) -> str:
        if not self.demo_names:
            raise ValueError(f"No demos found in {self.dataset_path}")
        demo_idx = (int(self.config.env.demo_offset) + int(episode_id)) % len(self.demo_names)
        demo_name = self.demo_names[demo_idx]
        with h5py.File(self.dataset_path, "r") as f:
            demo = f[f"data/{demo_name}"]
            reset_to(
                env,
                np.asarray(demo["states"][0]),
                model_xml=demo.attrs.get("model_file"),
                ep_meta=demo.attrs.get("ep_meta"),
            )
        return demo_name

    def _payload(self, env, lowdim_obs: dict[str, Any], step: int) -> dict[str, Any]:
        return make_observation_payload(
            env,
            lowdim_obs=lowdim_obs,
            step=step,
            camera_names=self.camera_names,
            camera_height=self.camera_height,
            camera_width=self.camera_width,
            stereo_baseline=self.stereo_baseline,
            state_dim=self.state_dim,
        )

    def _maybe_append_video_frame(self, frames: list[np.ndarray], obs: dict[str, Any], step: int) -> None:
        if not self.config.record.save_video:
            return
        skip = max(1, int(self.config.record.video_skip))
        if step % skip != 0:
            return
        frames.append(video_frame_from_payload(obs, self.camera_pairs))

    def _root(self) -> Path:
        return Path(self.config.save_root) / f"stseed-{self.config.env.seed}"

    def _write_episode(self, result: EpisodeResult) -> None:
        if not self.config.record.save_episode_jsonl:
            return
        path = self._root() / "metrics" / result.task_name / "episodes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")

    def _write_summary(self, metrics: dict[str, float]) -> None:
        if not self.config.record.save_metrics:
            return
        path = self._root() / "metrics" / self.task_name / "res.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _write_actions(self, episode_id: int, seed: int, records: list[dict[str, Any]]) -> None:
        path = self._root() / "actions" / self.task_name / f"seed_{seed}_episode_{episode_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_video(self, episode_id: int, seed: int, frames: list[np.ndarray], success: bool) -> None:
        path = self._root() / "visualization" / self.task_name / f"seed_{seed}_episode_{episode_id}_{success}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, frames, fps=20)

    def _summarize(self, results: list[EpisodeResult]) -> dict[str, float]:
        if not results:
            raise RuntimeError("No robomimic rollout results were produced.")
        success = sum(int(result.success) for result in results)
        horizons = [result.horizon for result in results]
        returns = [result.ret for result in results]
        return {
            "succ_num": float(success),
            "total_num": float(len(results)),
            "succ_rate": float(success / len(results)),
            "avg_horizon": float(np.mean(horizons)),
            "avg_return": float(np.mean(returns)),
        }


def video_frame_from_payload(obs: dict[str, Any], camera_pairs: list[list[str]]) -> np.ndarray:
    frames = []
    images = obs["images"]
    for left_key, right_key in camera_pairs:
        frames.append(np.asarray(images[payload_image_name(left_key)], dtype=np.uint8))
        frames.append(np.asarray(images[payload_image_name(right_key)], dtype=np.uint8))
    return np.concatenate(frames, axis=1)


def demo_metadata(dataset_path: str | Path, demo_name: str) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as f:
        demo = f[f"data/{demo_name}"]
        return {
            "num_samples": int(demo.attrs.get("num_samples", demo["states"].shape[0])),
            "model_file": bool(demo.attrs.get("model_file") is not None),
            "ep_meta": decode_attr(demo.attrs.get("ep_meta")) if demo.attrs.get("ep_meta") is not None else None,
        }


__all__ = [
    "EpisodeResult",
    "RobomimicEvaluator",
    "RobomimicPolicyClient",
    "demo_metadata",
    "select_action_steps",
    "video_frame_from_payload",
]

