#!/usr/bin/env bash
set -euo pipefail

session_name="${SESSION_NAME:-train_robotwin_topk2d}"
root_dir="/data/jsy/ffs"
payload="${root_dir}/scripts/run_train_robotwin_topk2d_payload.sh"

if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "tmux session '${session_name}' already exists."
  echo "Attach with: tmux attach -t ${session_name}"
  exit 0
fi

tmux new-session -d -s "${session_name}" "${payload}"
echo "Started tmux session '${session_name}'."
echo "Attach with: tmux attach -t ${session_name}"
echo "Log: ${root_dir}/train_robotwin_topk2d_tmux.log"
