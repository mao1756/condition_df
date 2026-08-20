#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv-runpod/bin/python"
MODULE="${DIRECTIONAL_MODULE:-mnist.diag_d0_jacobi_rb_quartile_directional_portable_continuation}"
RUNS_ROOT="${REPO_ROOT}/runs/experiment12_d0_jacobi_rb_quartile_directional_portable_continuation"
RUN_NAME="production-runpod-quartile-directional-continuation"
RUNTIME_ROOT="${REPO_ROOT}/runpod_runtime"

SOURCE_RUN="${REPO_ROOT}/runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication/20260808-203454_production-read-only-quartile-directional-adjudication-bootstrap-fix"
SPECIALIST_RUN="${REPO_ROOT}/runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist/20260807-132351_production-exact-quartile-specialist"
TIME_LOCAL_RUN="${REPO_ROOT}/runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication/20260807-005609_production-v3-time-local-adjudication"

cd "${REPO_ROOT}"
mkdir -p "${RUNS_ROOT}" "${RUNTIME_ROOT}"
COMMAND="${1:-status}"

# Status is deliberately lock-free and does not hash the 1.7-GiB handoff.
# It remains usable while the exclusive continuation process is running.
if [[ "${COMMAND}" == "status" ]]; then
  run_dir="$(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${RUN_NAME}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${run_dir}" ]]; then
    echo "portable continuation: not created"
  else
    echo "portable continuation: ${run_dir}"
    if [[ -f "${run_dir}/run_status.json" ]]; then
      cat "${run_dir}/run_status.json"
    else
      echo "status missing (relocation has not committed its first status yet)"
    fi
  fi
  exit 0
fi

LOG="${RUNTIME_ROOT}/directional-$(date -u +%Y%m%dT%H%M%SZ).log"
EXIT_RECORD="${RUNTIME_ROOT}/last_exit_code.txt"
LOCK="${RUNTIME_ROOT}/directional.lock"

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Another directional continuation launcher holds ${LOCK}." >&2
  exit 2
fi

exec > >(tee -a "${LOG}") 2>&1
trap 'code=$?; printf "%s\n" "${code}" > "${EXIT_RECORD}"; echo "launcher exit code: ${code}"; exit "${code}"' EXIT

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run tools/runpod_directional/install_env.sh first." >&2
  exit 2
fi

# Preserve the predecessor runtime contract.  In particular, do not set or
# unset backend flags here; fail if the selected pod inherited a mismatch.
export PYTHONPATH="${REPO_ROOT}"
export PYTHONUNBUFFERED=1
"${PYTHON}" "${SCRIPT_DIR}/verify_bundle.py" \
  --root "${REPO_ROOT}" --check-runtime

latest_run() {
  local value
  value="$(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${RUN_NAME}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  printf '%s' "${value}"
}

read_status_field() {
  local run_dir="$1"
  local field="$2"
  "${PYTHON}" - "${run_dir}/run_status.json" "${field}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    print("")
else:
    value = json.loads(path.read_text(encoding="utf-8")).get(sys.argv[2], "")
    print(value)
PY
}

run_relocate() {
  "${PYTHON}" -m "${MODULE}" \
    --runs-root "${RUNS_ROOT}" \
    --run-name "${RUN_NAME}" \
    --device cuda \
    --stage relocate \
    --source-adjudication-run-dir "${SOURCE_RUN}" \
    --parent-quartile-specialist-run-dir "${SPECIALIST_RUN}" \
    --parent-time-local-run-dir "${TIME_LOCAL_RUN}" \
    --require-gate relocate
}

run_resumed_stage() {
  local stage="$1"
  local required_gate="${stage}"
  local run_dir
  run_dir="$(latest_run)"
  if [[ -z "${run_dir}" ]]; then
    echo "No portable continuation exists; run relocate first." >&2
    return 2
  fi
  if [[ "${stage}" == "report" ]]; then
    required_gate="none"
  fi
  "${PYTHON}" -m "${MODULE}" \
    --device cuda \
    --stage "${stage}" \
    --resume-run-dir "${run_dir}" \
    --source-adjudication-run-dir "${SOURCE_RUN}" \
    --parent-quartile-specialist-run-dir "${SPECIALIST_RUN}" \
    --parent-time-local-run-dir "${TIME_LOCAL_RUN}" \
    --require-gate "${required_gate}"
}

continue_stages() {
  local run_dir decision state stage
  for _ in 1 2 3 4 5 6; do
    run_dir="$(latest_run)"
    if [[ -z "${run_dir}" ]]; then
      run_relocate
      continue
    fi
    decision="$(read_status_field "${run_dir}" decision)"
    state="$(read_status_field "${run_dir}" state)"
    stage="$(read_status_field "${run_dir}" stage)"
    echo "continuation status: stage=${stage} state=${state} decision=${decision}"
    case "${decision}" in
      ""|interrupted_relocate|running_relocate)
        run_resumed_stage relocate
        ;;
      ready_for_fittrace|interrupted_fittrace|running_fittrace)
        run_resumed_stage fittrace
        ;;
      ready_for_nominate|interrupted_nominate|running_nominate)
        run_resumed_stage nominate
        ;;
      ready_for_adjudicate|interrupted_adjudicate|running_adjudicate)
        run_resumed_stage adjudicate
        ;;
      ready_for_report|interrupted_report|running_report)
        run_resumed_stage report
        ;;
      portable_continuation_complete|report_complete)
        echo "Portable continuation is complete."
        return 0
        ;;
      *)
        if [[ "${stage}" == "adjudicate" \
          && ( "${state}" == "complete" || "${state}" == "valid_scientific_stop" ) ]]; then
          run_resumed_stage report
          continue
        fi
        if [[ "${stage}" == "report" \
          && ( "${state}" == "complete" || "${state}" == "valid_scientific_stop" ) ]]; then
          echo "Portable continuation is complete: ${decision}"
          return 0
        fi
        echo "Failing closed on unrecognized/nonpassing status: ${decision}" >&2
        return 1
        ;;
    esac
  done
  echo "Continuation exceeded its fixed stage-advance count." >&2
  return 1
}

case "${COMMAND}" in
  relocate)
    if [[ -n "$(latest_run)" ]]; then
      echo "A portable continuation already exists; refusing a second relocate." >&2
      exit 2
    fi
    run_relocate
    ;;
  fittrace|nominate|adjudicate|report)
    run_resumed_stage "${COMMAND}"
    ;;
  continue)
    continue_stages
    ;;
  *)
    echo "usage: $0 {relocate|fittrace|nominate|adjudicate|report|continue|status}" >&2
    exit 2
    ;;
esac
