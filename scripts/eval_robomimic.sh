#!/bin/bash
set -e

ROOT_DIR=${ROOT_DIR:-/data/jsy/ffs}
CONFIG=${CONFIG:-configs/robomimic_square_eval.yaml}
CLIENT_PYTHON=${CLIENT_PYTHON:-python3}
SAVE_ROOT=${SAVE_ROOT:-outputs/robomimic_square_diffusion_eval}
DATASET=${DATASET:-/data/jsy/robomimic/datasets/square/ph/stereo_image_v15.hdf5}
CKPT=${CKPT:-outputs/robomimic_square_diffusion_aligned/latest.pt}
FFS_CONFIG_PATH=${FFS_CONFIG_PATH:-}
PORT=${PORT:-29068}
TEST_NUM=${TEST_NUM:-50}
HORIZON=${HORIZON:-400}
SEED=${SEED:-0}
MUJOCO_GL=${MUJOCO_GL:-egl}
RENDER_GPU_DEVICE_ID=${RENDER_GPU_DEVICE_ID:-}
EXECUTE_CHUNK_STEPS=${EXECUTE_CHUNK_STEPS:-}
SAVE_VIDEO=${SAVE_VIDEO:-}
DRY_RUN=${DRY_RUN:-}

cd "$ROOT_DIR"

cmd=("$CLIENT_PYTHON" -m ffs.evaluation.robomimic.cli
    --config "$CONFIG"
    --save-root "$SAVE_ROOT"
    --dataset "$DATASET"
    --checkpoint "$CKPT"
    --port "$PORT"
    --n-rollouts "$TEST_NUM"
    --horizon "$HORIZON"
    --seed "$SEED"
    --mujoco-gl "$MUJOCO_GL")

if [ -n "$RENDER_GPU_DEVICE_ID" ]; then
    cmd+=(--render-gpu-device-id "$RENDER_GPU_DEVICE_ID")
fi
if [ -n "$FFS_CONFIG_PATH" ]; then
    cmd+=(--config-path "$FFS_CONFIG_PATH")
fi
if [ -n "$EXECUTE_CHUNK_STEPS" ]; then
    cmd+=(--execute-chunk-steps "$EXECUTE_CHUNK_STEPS")
fi
if [ "$SAVE_VIDEO" = "1" ] || [ "$SAVE_VIDEO" = "true" ]; then
    cmd+=(--save-video)
fi
if [ "$DRY_RUN" = "1" ] || [ "$DRY_RUN" = "true" ]; then
    cmd+=(--dry-run)
fi

NUMBA_DISABLE_JIT=1 PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "${cmd[@]}"
