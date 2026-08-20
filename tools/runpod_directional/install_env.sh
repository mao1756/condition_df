#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="${RUNPOD_WORKSPACE_ROOT:-/workspace}"
UV_BIN="${WORKSPACE_ROOT}/bin/uv"
VENV="${REPO_ROOT}/.venv-runpod"

mkdir -p "${WORKSPACE_ROOT}/bin" "${WORKSPACE_ROOT}/.cache/uv" \
  "${WORKSPACE_ROOT}/.uv/python"
export UV_CACHE_DIR="${WORKSPACE_ROOT}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${WORKSPACE_ROOT}/.uv/python"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "Installing uv into persistent storage..."
  export UV_INSTALL_DIR="${WORKSPACE_ROOT}/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

"${UV_BIN}" python install 3.14.4
if [[ ! -x "${VENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.14.4 --seed "${VENV}"
fi

PYTHON="${VENV}/bin/python"
PYTHON_VERSION="$("${PYTHON}" -c 'import platform; print(platform.python_version())')"
if [[ "${PYTHON_VERSION}" != "3.14.4" ]]; then
  echo "Existing ${VENV} uses Python ${PYTHON_VERSION}, expected 3.14.4." >&2
  echo "Remove that handoff-only environment explicitly, then rerun setup." >&2
  exit 1
fi
"${PYTHON}" -m pip install --upgrade pip==26.0.1
"${PYTHON}" -m pip install -r "${REPO_ROOT}/requirements-runpod-directional.txt"

# Install the exact CUDA wheels after their ordinary PyPI dependencies.  The
# no-deps install prevents a different index from silently replacing a pin.
"${PYTHON}" -m pip install --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128

unset PYTHONPATH
"${PYTHON}" "${SCRIPT_DIR}/verify_bundle.py" \
  --root "${REPO_ROOT}" --check-runtime

echo
echo "RunPod directional runtime is ready: ${VENV}"
echo "The predecessor runtime contract intentionally leaves"
echo "CUBLAS_WORKSPACE_CONFIG unset and preserves the Torch backend defaults."
