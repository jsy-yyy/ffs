#!/usr/bin/env bash
set -uo pipefail

cd /data/jsy/ffs || exit 1
ulimit -n 65535 || true

run_id="$(date +%Y%m%d_%H%M%S)"
log_path="train_robotwin_topk2d_tmux.log"
rank_log_dir="train_robotwin_topk2d_rank_logs/${run_id}"
mkdir -p "${rank_log_dir}"

if [[ -f "${log_path}" ]]; then
  mv "${log_path}" "${log_path}.${run_id}.bak"
fi

exec > >(tee -a "${log_path}") 2>&1

echo "[START] $(date -Is) host=$(hostname) pid=$$ ulimit_n=$(ulimit -n) run_id=${run_id}"
echo "[RANK_LOG_DIR] ${rank_log_dir}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore:Default grid_sample and affine_grid behavior has changed:UserWarning,ignore:pkg_resources is deprecated as an API:UserWarning"

/home/zmz/miniconda3/envs/ffs/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  --log-dir "${rank_log_dir}" --tee 3 \
  scripts/train.py \
  --config configs/waft_rdt_robotwin_topk2d_hdf5.yaml

rc=$?
echo "[EXIT] $(date -Is) rc=${rc}"
exec bash
