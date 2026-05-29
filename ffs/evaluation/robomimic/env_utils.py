from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


IMAGE_PREFIX = "observation.images."


def apply_runtime_env(numba_disable_jit: bool = True, mujoco_gl: str | None = "egl") -> None:
    if numba_disable_jit:
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    if mujoco_gl:
        os.environ.setdefault("MUJOCO_GL", mujoco_gl)


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def load_env_meta(dataset_path: str | Path) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as f:
        env_args = f["data"].attrs["env_args"]
        return json.loads(decode_attr(env_args))


def sorted_demo_names(dataset_path: str | Path) -> list[str]:
    with h5py.File(dataset_path, "r") as f:
        return sorted(f["data"].keys(), key=lambda name: int(name.split("_")[-1]))


def payload_image_name(feature_key: str) -> str:
    return feature_key[len(IMAGE_PREFIX) :] if feature_key.startswith(IMAGE_PREFIX) else feature_key


def camera_base_name(feature_key: str) -> str:
    name = payload_image_name(feature_key)
    for suffix in ("_left", "_right"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Expected stereo image key ending in _left or _right, got {feature_key!r}")


def camera_names_from_pairs(camera_pairs: list[list[str]]) -> list[str]:
    names: list[str] = []
    for left_key, right_key in camera_pairs:
        left_name = camera_base_name(left_key)
        right_name = camera_base_name(right_key)
        if left_name != right_name:
            raise ValueError(
                f"Stereo pair does not share a camera base name: {left_key!r}, {right_key!r}"
            )
        if left_name not in names:
            names.append(left_name)
    if not names:
        raise ValueError("No camera pairs configured for robomimic evaluation.")
    return names


def make_robosuite_env(
    env_meta: dict[str, Any],
    camera_names: list[str],
    camera_height: int,
    camera_width: int,
    render_gpu_device_id: int | None = None,
):
    import robosuite

    env_kwargs = copy.deepcopy(env_meta["env_kwargs"])
    env_kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names=list(camera_names),
        camera_heights=int(camera_height),
        camera_widths=int(camera_width),
        ignore_done=True,
    )
    if render_gpu_device_id is not None:
        env_kwargs["render_gpu_device_id"] = int(render_gpu_device_id)

    for key in (
        "env_name",
        "camera_height",
        "camera_width",
        "render",
        "render_offscreen",
        "use_image_obs",
        "use_depth_obs",
    ):
        env_kwargs.pop(key, None)

    version_parts = getattr(robosuite, "__version__", "1.4.0").split(".")[:2]
    try:
        major, minor = (int(version_parts[0]), int(version_parts[1]))
    except Exception:
        major, minor = (1, 4)
    if major == 1 and minor < 5:
        env_kwargs.pop("lite_physics", None)

    env_version = env_meta.get("env_version")
    if env_version is not None and env_version != getattr(robosuite, "__version__", None):
        print(
            "WARNING: dataset robosuite version is "
            f"{env_version}, but installed version is {robosuite.__version__}.",
            flush=True,
        )
    return robosuite.make(env_meta["env_name"], **env_kwargs)


def reset_to(env, state: np.ndarray, model_xml: Any | None = None, ep_meta: Any | None = None) -> None:
    if model_xml is not None:
        if ep_meta is not None and hasattr(env, "set_ep_meta"):
            env.set_ep_meta(json.loads(decode_attr(ep_meta)))
        elif hasattr(env, "unset_ep_meta"):
            env.unset_ep_meta()

        env.reset()
        if hasattr(env, "edit_model_xml"):
            xml = env.edit_model_xml(decode_attr(model_xml))
        else:
            xml = decode_attr(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

    env.sim.set_state_from_flattened(state)
    env.sim.forward()


def get_lowdim_obs(env) -> dict[str, np.ndarray]:
    if not hasattr(env, "_get_observations"):
        raise AttributeError("robosuite env does not expose _get_observations()")
    return env._get_observations()


def get_camera_local_x_delta(sim, cam_id: int, distance: float) -> np.ndarray:
    world_right = sim.data.cam_xmat[cam_id].reshape(3, 3)[:, 0]
    body_id = int(sim.model.cam_bodyid[cam_id])
    if body_id >= 0:
        body_rot = sim.data.body_xmat[body_id].reshape(3, 3)
        local_right = body_rot.T.dot(world_right)
    else:
        local_right = world_right
    return local_right * distance


def render_shifted_camera(env, camera_name: str, height: int, width: int, offset: np.ndarray) -> np.ndarray:
    sim = env.sim
    cam_id = sim.model.camera_name2id(camera_name)
    original_pos = sim.model.cam_pos[cam_id].copy()
    sim.model.cam_pos[cam_id] = original_pos + offset
    sim.forward()
    image = sim.render(height=int(height), width=int(width), camera_name=camera_name)[::-1].copy()
    sim.model.cam_pos[cam_id] = original_pos
    sim.forward()
    return image


def render_stereo_images(
    env,
    camera_names: list[str],
    height: int,
    width: int,
    baseline: float,
) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    half = float(baseline) / 2.0
    for camera_name in camera_names:
        cam_id = env.sim.model.camera_name2id(camera_name)
        local_half_delta = get_camera_local_x_delta(env.sim, cam_id, half)
        images[f"{camera_name}_left"] = render_shifted_camera(
            env, camera_name, height, width, -local_half_delta
        )
        images[f"{camera_name}_right"] = render_shifted_camera(
            env, camera_name, height, width, local_half_delta
        )
    return images


def state_from_robosuite_obs(obs: dict[str, Any], state_dim: int) -> np.ndarray:
    object_obs = obs["object"] if "object" in obs else obs["object-state"]

    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(object_obs, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    if state.shape != (state_dim,):
        raise ValueError(f"Expected robomimic state shape ({state_dim},), got {state.shape}.")
    return state


def make_observation_payload(
    env,
    lowdim_obs: dict[str, Any],
    step: int,
    camera_names: list[str],
    camera_height: int,
    camera_width: int,
    stereo_baseline: float,
    state_dim: int,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "state": state_from_robosuite_obs(lowdim_obs, state_dim),
        "images": render_stereo_images(
            env,
            camera_names=camera_names,
            height=camera_height,
            width=camera_width,
            baseline=stereo_baseline,
        ),
    }


def check_success(env) -> bool:
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    if hasattr(env, "is_success"):
        value = env.is_success()
        if isinstance(value, dict):
            return bool(value.get("task", False))
        return bool(value)
    return False


__all__ = [
    "apply_runtime_env",
    "camera_names_from_pairs",
    "check_success",
    "decode_attr",
    "get_lowdim_obs",
    "load_env_meta",
    "make_observation_payload",
    "make_robosuite_env",
    "payload_image_name",
    "render_stereo_images",
    "reset_to",
    "sorted_demo_names",
    "state_from_robosuite_obs",
]
