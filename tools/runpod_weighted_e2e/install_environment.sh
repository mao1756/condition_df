#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-runpod-weighted}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REPO_ROOT}/tools/runpod_weighted_e2e/requirements.lock"

python - <<'PY'
import torch
import torchvision
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected RunPod image")
print({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
})
PY
