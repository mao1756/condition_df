#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${RUN_DIR}.tar.zst}"
FINAL_ACTION="${RUNPOD_FINAL_ACTION:-delete}"
EXPORT_VERIFIED="${EXPORT_VERIFIED:-0}"
WATCHDOG_PID="${WATCHDOG_PID:-}"

if [[ -n "${WATCHDOG_PID}" ]]; then
  kill "${WATCHDOG_PID}" 2>/dev/null || true
fi
sync || true

if [[ "${EXPORT_VERIFIED}" != "1" ]]; then
  echo "Durable export was not verified; stopping rather than deleting the Pod" >&2
  FINAL_ACTION="stop"
fi

mkdir -p "${RUN_DIR}"
export ARCHIVE_PATH EXPORT_VERIFIED FINAL_ACTION
python3 - "${RUN_DIR}/runpod_finalization.json" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "d0-jacobi-rb-path-weighted-runpod-finalization-v1",
    "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "pod_id": os.environ.get("RUNPOD_POD_ID"),
    "archive_path": os.environ.get("ARCHIVE_PATH"),
    "export_verified": int(os.environ.get("EXPORT_VERIFIED", "0")),
    "requested_final_action": os.environ.get("RUNPOD_FINAL_ACTION", "delete"),
    "effective_final_action": os.environ.get("FINAL_ACTION", "stop"),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
temporary.replace(path)
PY

if [[ "${FINAL_ACTION}" == "delete" ]]; then
  "${REPO_ROOT}/tools/runpod_weighted_e2e/pod_lifecycle.sh" delete || \
    "${REPO_ROOT}/tools/runpod_weighted_e2e/pod_lifecycle.sh" stop
else
  "${REPO_ROOT}/tools/runpod_weighted_e2e/pod_lifecycle.sh" stop
fi
