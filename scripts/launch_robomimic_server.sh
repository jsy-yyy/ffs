#!/bin/bash
set -e

ROOT_DIR=${ROOT_DIR:-/data/jsy/ffs}
CONFIG=${CONFIG:-configs/robomimic_square_eval.yaml}
SERVER_PYTHON=${SERVER_PYTHON:-python3}
PORT=${PORT:-29068}
DEVICE=${DEVICE:-cuda:0}
CKPT=${CKPT:-outputs/robomimic_square_rdt/latest.pt}
FFS_CONFIG_PATH=${FFS_CONFIG_PATH:-}
SAMPLE_INIT=${SAMPLE_INIT:-}
DISPARITY_ABLATION=${DISPARITY_ABLATION:-}

cd "$ROOT_DIR"

cmd=("$SERVER_PYTHON" -m ffs.evaluation.robomimic.server
    --config "$CONFIG"
    --port "$PORT"
    --device "$DEVICE"
    --checkpoint "$CKPT")

if [ -n "$FFS_CONFIG_PATH" ]; then
    cmd+=(--config-path "$FFS_CONFIG_PATH")
fi
if [ -n "$SAMPLE_INIT" ]; then
    cmd+=(--sample-init "$SAMPLE_INIT")
fi
if [ -n "$DISPARITY_ABLATION" ]; then
    cmd+=(--disparity-ablation "$DISPARITY_ABLATION")
fi

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "${cmd[@]}"

