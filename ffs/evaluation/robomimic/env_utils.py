from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import h5py
import numpy as np


IMAGE_PREFIX = "observation.images."
DEFAULT_ROBOMIMIC_ROOT = "/data/jsy/robomimic"


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
    from copy import deepcopy

    _prefer_local_robomimic()
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.env_utils as EnvUtils

    env_kwargs = deepcopy(env_meta["env_kwargs"])
    env_kwargs["camera_names"] = list(camera_names)
    env_kwargs["camera_heights"] = int(camera_height)
    env_kwargs["camera_widths"] = int(camera_width)
    for key in ("stereo_baseline_m", "stereo_camera_names", "stereo_camera_info"):
        if key in env_meta:
            env_kwargs[key] = env_meta[key]
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
    import robosuite

    version_parts = getattr(robosuite, "__version__", "1.4.0").split(".")[:2]
    try:
        major, minor = (int(version_parts[0]), int(version_parts[1]))
    except Exception:
        major, minor = (1, 4)
    if major == 1 and minor < 5:
        env_kwargs.pop("lite_physics", None)
        env_kwargs["controller_configs"] = _legacy_robosuite_controller_config(
            env_kwargs.get("controller_configs")
        )

    stereo_image_keys = [f"{camera_name}_{side}_image" for camera_name in camera_names for side in ("left", "right")]
    _initialize_robomimic_obs_utils(ObsUtils, stereo_image_keys, use_depth_obs=False)
    env = EnvUtils.create_env(
        env_type=EnvUtils.get_env_type(env_meta=env_meta),
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=False,
        **env_kwargs,
    )
    EnvUtils.check_env_version(env, env_meta)
    return env


def _legacy_robosuite_controller_config(controller_configs: Any) -> Any:
    if not isinstance(controller_configs, dict):
        return controller_configs
    body_parts = controller_configs.get("body_parts")
    if not isinstance(body_parts, dict):
        return controller_configs
    right = body_parts.get("right")
    if not isinstance(right, dict):
        return controller_configs

    legacy = dict(right)
    legacy.pop("gripper", None)
    legacy.pop("input_ref_frame", None)
    if "damping" in legacy and "damping_ratio" not in legacy:
        legacy["damping_ratio"] = legacy.pop("damping")
    if "damping_limits" in legacy and "damping_ratio_limits" not in legacy:
        legacy["damping_ratio_limits"] = legacy.pop("damping_limits")
    return legacy


def _initialize_robomimic_obs_utils(ObsUtils, rgb_keys: list[str], use_depth_obs: bool) -> None:
    obs_spec: dict[str, dict[str, list[str]]] = {
        "obs": {
            "low_dim": [],
            "rgb": list(rgb_keys),
        }
    }
    if use_depth_obs:
        obs_spec["obs"]["depth"] = [key.replace("_image", "_depth") for key in rgb_keys]
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_spec)


def _prefer_local_robomimic() -> None:
    robomimic_root = Path(os.environ.get("ROBOMIMIC_ROOT", DEFAULT_ROBOMIMIC_ROOT))
    if (robomimic_root / "robomimic").is_dir() and str(robomimic_root) not in sys.path:
        sys.path.insert(0, str(robomimic_root))
    if "robomimic.utils.lang_utils" not in sys.modules:
        lang_utils = types.ModuleType("robomimic.utils.lang_utils")
        lang_utils.LANG_EMB_OBS_KEY = "lang_emb"
        lang_utils.get_lang_emb = lambda lang: None
        lang_utils.get_lang_emb_shape = lambda: []
        sys.modules["robomimic.utils.lang_utils"] = lang_utils


def unwrap_robosuite_env(env):
    current = env
    while not hasattr(current, "sim") and hasattr(current, "env"):
        current = current.env
    if not hasattr(current, "sim"):
        raise AttributeError("Could not find underlying robosuite env with a sim attribute.")
    return current


def get_lowdim_obs(env) -> dict[str, np.ndarray]:
    if hasattr(env, "get_observation"):
        return env.get_observation()
    core_env = unwrap_robosuite_env(env)
    if not hasattr(core_env, "_get_observations"):
        raise AttributeError("robosuite env does not expose _get_observations()")
    return core_env._get_observations()


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
    sim = unwrap_robosuite_env(env).sim
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
    sim = unwrap_robosuite_env(env).sim
    for camera_name in camera_names:
        cam_id = sim.model.camera_name2id(camera_name)
        local_half_delta = get_camera_local_x_delta(sim, cam_id, half)
        images[f"{camera_name}_left"] = render_shifted_camera(
            env, camera_name, height, width, -local_half_delta
        )
        images[f"{camera_name}_right"] = render_shifted_camera(
            env, camera_name, height, width, local_half_delta
        )
    return images


def state_from_robosuite_obs(obs: dict[str, Any], state_dim: int) -> np.ndarray:
    if state_dim == 16:
        return np.concatenate(
            [
                np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

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
    # if state.shape != (state_dim,):
    #     raise ValueError(f"Expected robomimic state shape ({state_dim},), got {state.shape}.")
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
    images: dict[str, np.ndarray] = {}
    expected_image_names = [
        f"{camera_name}_{side}"
        for camera_name in camera_names
        for side in ("left", "right")
    ]
    for image_name in expected_image_names:
        env_key = f"{image_name}_image"
        if env_key in lowdim_obs:
            images[image_name] = np.asarray(lowdim_obs[env_key]).copy()
    if len(images) != len(expected_image_names):
        images = render_stereo_images(
            env,
            camera_names=camera_names,
            height=camera_height,
            width=camera_width,
            baseline=stereo_baseline,
        )
    return {
        "step": int(step),
        "state": state_from_robosuite_obs(lowdim_obs, state_dim),
        "images": images,
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
    "state_from_robosuite_obs",
    "unwrap_robosuite_env",
]
