#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
DATA_DIR="${DATA_DIR:-/workspace/mnist_data}"
DEVICE="${DEVICE:-cuda:0}"
SMALL_UPDATES="${SMALL_UPDATES:-12000}"
LARGE_UPDATES="${LARGE_UPDATES:-12000}"
TRAIN_PATHS="${TRAIN_PATHS:-64}"
VALIDATION_PATHS="${VALIDATION_PATHS:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-250}"
MNIST_INDEX="${MNIST_INDEX:-0}"
MOBILITY_FLOOR="${MOBILITY_FLOOR:-1e-4}"
HARD_WALL_SECONDS="${HARD_WALL_SECONDS:-21600}"
WATCHDOG_GRACE_SECONDS="${WATCHDOG_GRACE_SECONDS:-900}"
RESULTS_URI="${RESULTS_URI:-}"
RUNPOD_RESULTS_DURABLE="${RUNPOD_RESULTS_DURABLE:-0}"
RUNPOD_FINAL_ACTION="${RUNPOD_FINAL_ACTION:-delete}"
export REPO_ROOT RUN_DIR RUNPOD_RESULTS_DURABLE RUNPOD_FINAL_ACTION

mkdir -p "${RUN_DIR}"

# Independent GPU-cost fuse: even if Python or this shell wedges, the Pod is stopped.
(
  sleep "$((HARD_WALL_SECONDS + WATCHDOG_GRACE_SECONDS))"
  echo "RunPod hard-wall watchdog fired" >&2
  bash "${REPO_ROOT}/tools/runpod_weighted_e2e/pod_lifecycle.sh" stop
) &
WATCHDOG_PID=$!
export WATCHDOG_PID

EXPORT_VERIFIED=0
ARCHIVE_PATH="${RUN_DIR}.tar.zst"
export EXPORT_VERIFIED ARCHIVE_PATH
finalized=0
finalize_once() {
  if [[ "${finalized}" == "0" ]]; then
    finalized=1
    export EXPORT_VERIFIED ARCHIVE_PATH WATCHDOG_PID
    bash "${REPO_ROOT}/tools/runpod_weighted_e2e/finalize.sh" || true
  fi
}
trap finalize_once EXIT INT TERM

cd "${REPO_ROOT}"
bash "${REPO_ROOT}/tools/runpod_weighted_e2e/install_environment.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv-runpod-weighted/bin/activate"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Verify protected historical source by canonical source content.  The frozen
# byte-level inventory includes a mixed-newline file, so exact byte hashes are
# not portable across Git/ZIP checkouts.  Canonical hashing permits newline-only
# conversion and still rejects every other source edit.
python -m mnist.d0_jacobi_rb_runpod_source_integrity \
  --repository-root "${REPO_ROOT}"

# Ruff import sorting is formatting-only and safe to repair in the disposable
# RunPod checkout.  The second command remains the strict E/F/I gate.
python -m ruff check --select I --fix \
  mnist/d0_jacobi_rb_path_weighted_loss.py \
  mnist/d0_jacobi_rb_runpod_source_integrity.py \
  mnist/d0_jacobi_rb_nvrtc_compat.py \
  mnist/d0_jacobi_rb_global_large.py \
  mnist/d0_jacobi_rb_candidate_training_cache.py \
  mnist/d0_jacobi_rb_path_weighted_training.py \
  mnist/diag_d0_jacobi_rb_path_weighted_capacity_e2e.py \
  tests/test_d0_jacobi_rb_path_weighted_loss.py \
  tests/test_d0_jacobi_rb_nvrtc_compat.py \
  tests/test_d0_jacobi_rb_global_large.py \
  tests/test_d0_jacobi_rb_candidate_training_cache.py \
  tests/test_diag_d0_jacobi_rb_path_weighted_capacity_e2e.py

python -m ruff check \
  mnist/d0_jacobi_rb_path_weighted_loss.py \
  mnist/d0_jacobi_rb_runpod_source_integrity.py \
  mnist/d0_jacobi_rb_nvrtc_compat.py \
  mnist/d0_jacobi_rb_global_large.py \
  mnist/d0_jacobi_rb_candidate_training_cache.py \
  mnist/d0_jacobi_rb_path_weighted_training.py \
  mnist/diag_d0_jacobi_rb_path_weighted_capacity_e2e.py \
  tests/test_d0_jacobi_rb_path_weighted_loss.py \
  tests/test_d0_jacobi_rb_nvrtc_compat.py \
  tests/test_d0_jacobi_rb_global_large.py \
  tests/test_d0_jacobi_rb_candidate_training_cache.py \
  tests/test_diag_d0_jacobi_rb_path_weighted_capacity_e2e.py

python -m pytest -q \
  tests/test_d0_jacobi_rb_path_weighted_loss.py \
  tests/test_d0_jacobi_rb_nvrtc_compat.py \
  tests/test_d0_jacobi_rb_global_large.py \
  tests/test_d0_jacobi_rb_candidate_training_cache.py \
  tests/test_diag_d0_jacobi_rb_path_weighted_capacity_e2e.py
python -m pytest -q tests/test_eulerian_jacobi_ddpm_candidate.py \
  -k 'candidate_dispatch_uses_only_candidate_prepare_and_enqueue or candidate_phase_preserves_orientation_ids_pair_totals_and_simplex or candidate_forward_phase_supports_k512 or candidate_eager_prefixes_share_uniform_ids_and_scale_exposure'

set +e
timeout --signal=TERM --kill-after=120 "${HARD_WALL_SECONDS}" \
  python -m mnist.diag_d0_jacobi_rb_path_weighted_capacity_e2e run \
    --run-dir "${RUN_DIR}" \
    --device "${DEVICE}" \
    --data-dir "${DATA_DIR}" \
    --mnist-index "${MNIST_INDEX}" \
    --train-paths "${TRAIN_PATHS}" \
    --validation-paths "${VALIDATION_PATHS}" \
    --small-updates "${SMALL_UPDATES}" \
    --large-updates "${LARGE_UPDATES}" \
    --batch-size "${BATCH_SIZE}" \
    --validation-interval "${VALIDATION_INTERVAL}" \
    --mobility-floor "${MOBILITY_FLOOR}" \
    --hard-wall-seconds "${HARD_WALL_SECONDS}"
experiment_status=$?
set -e

# A failed scientific run is still valuable if its terminal bundle is intact.
python -m mnist.diag_d0_jacobi_rb_path_weighted_capacity_e2e verify \
  --run-dir "${RUN_DIR}" || true

if command -v zstd >/dev/null 2>&1; then
  tar --zstd -cf "${ARCHIVE_PATH}" -C "$(dirname "${RUN_DIR}")" "$(basename "${RUN_DIR}")"
else
  ARCHIVE_PATH="${RUN_DIR}.tar.gz"
  export ARCHIVE_PATH
  tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${RUN_DIR}")" "$(basename "${RUN_DIR}")"
fi
sha256sum "${ARCHIVE_PATH}" > "${ARCHIVE_PATH}.sha256"
local_archive_sha=$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')
sync
readback_archive_sha=$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')
ARCHIVE_LOCAL_VERIFIED=0
if [[ "${local_archive_sha}" == "${readback_archive_sha}" ]]; then
  ARCHIVE_LOCAL_VERIFIED=1
fi

if [[ -n "${RESULTS_URI}" ]]; then
  if [[ "${RESULTS_URI}" != s3://* ]]; then
    echo "RESULTS_URI currently supports only s3:// destinations" >&2
  elif ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI is required for RESULTS_URI export" >&2
  else
    destination="${RESULTS_URI%/}/$(basename "${ARCHIVE_PATH}")"
    aws s3 cp "${ARCHIVE_PATH}" "${destination}"
    remote_sha=$(aws s3 cp "${destination}" - | sha256sum | awk '{print $1}')
    if [[ "${remote_sha}" == "${local_archive_sha}" && "${ARCHIVE_LOCAL_VERIFIED}" == "1" ]]; then
      aws s3 cp "${ARCHIVE_PATH}.sha256" "${destination}.sha256"
      EXPORT_VERIFIED=1
    fi
  fi
elif [[ "${RUNPOD_RESULTS_DURABLE}" == "1" && "${RUN_DIR}" == /workspace/* && "${ARCHIVE_LOCAL_VERIFIED}" == "1" ]]; then
  # RUNPOD_RESULTS_DURABLE=1 is an explicit assertion that /workspace is backed
  # by a network volume which survives Pod deletion.
  EXPORT_VERIFIED=1
else
  echo "No verified network volume or S3 destination; preserving the Pod volume by stopping" >&2
fi
export EXPORT_VERIFIED

exit "${experiment_status}"
