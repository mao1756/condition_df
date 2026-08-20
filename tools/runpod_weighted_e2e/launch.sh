#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results/jacobi-path-weighted-capacity}"
RUN_NAME="${RUN_NAME:-run-$(date -u +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RESULTS_ROOT}/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}.log}"
LOCK_DIR="${RUN_DIR}.launch-lock"
RUNPOD_FINAL_ACTION="${RUNPOD_FINAL_ACTION:-delete}"
RUNPOD_RESULTS_DURABLE="${RUNPOD_RESULTS_DURABLE:-0}"
RUNPOD_DRY_RUN="${RUNPOD_DRY_RUN:-0}"

if [[ "${RUNPOD_DRY_RUN}" != "1" && -z "${RUNPOD_POD_ID:-}" ]]; then
  echo "RUNPOD_POD_ID is required for an unattended RunPod launch" >&2
  exit 1
fi
if [[ "${RUNPOD_FINAL_ACTION}" == "delete" && "${RUNPOD_RESULTS_DURABLE}" != "1" ]]; then
  echo "Refusing unattended Pod deletion without RUNPOD_RESULTS_DURABLE=1" >&2
  echo "Use an attached network volume at /workspace or a verified RESULTS_URI export." >&2
  exit 1
fi
mkdir -p "$(dirname "${RUN_DIR}")"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "This run has already been launched: ${RUN_DIR}" >&2
  exit 1
fi

export REPO_ROOT RUN_DIR
nohup setsid env \
  REPO_ROOT="${REPO_ROOT}" \
  RUN_DIR="${RUN_DIR}" \
  DATA_DIR="${DATA_DIR:-/workspace/mnist_data}" \
  DEVICE="${DEVICE:-cuda:0}" \
  SMALL_UPDATES="${SMALL_UPDATES:-12000}" \
  LARGE_UPDATES="${LARGE_UPDATES:-12000}" \
  TRAIN_PATHS="${TRAIN_PATHS:-64}" \
  VALIDATION_PATHS="${VALIDATION_PATHS:-32}" \
  BATCH_SIZE="${BATCH_SIZE:-32}" \
  VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-250}" \
  MNIST_INDEX="${MNIST_INDEX:-0}" \
  MOBILITY_FLOOR="${MOBILITY_FLOOR:-1e-4}" \
  HARD_WALL_SECONDS="${HARD_WALL_SECONDS:-21600}" \
  WATCHDOG_GRACE_SECONDS="${WATCHDOG_GRACE_SECONDS:-900}" \
  RESULTS_URI="${RESULTS_URI:-}" \
  RUNPOD_RESULTS_DURABLE="${RUNPOD_RESULTS_DURABLE}" \
  RUNPOD_FINAL_ACTION="${RUNPOD_FINAL_ACTION}" \
  RUNPOD_DRY_RUN="${RUNPOD_DRY_RUN}" \
  bash "${REPO_ROOT}/tools/runpod_weighted_e2e/worker.sh" \
  >"${LOG_PATH}" 2>&1 </dev/null &
worker_pid=$!

echo "Run launched in a detached session."
echo "  PID: ${worker_pid}"
echo "  Run directory: ${RUN_DIR}"
echo "  Log: ${LOG_PATH}"
echo "  Follow: tail -f '${LOG_PATH}'"
