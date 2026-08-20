#!/usr/bin/env bash
set -euo pipefail

action="${1:?usage: pod_lifecycle.sh <stop|delete>}"
pod_id="${RUNPOD_POD_ID:-}"
api_key="${RUNPOD_API_KEY:-}"
durable="${RUNPOD_RESULTS_DURABLE:-0}"

case "${action}" in
  stop|delete) ;;
  *) echo "unsupported Pod action: ${action}" >&2; exit 2 ;;
esac

if [[ "${RUNPOD_DRY_RUN:-0}" == "1" ]]; then
  echo "DRY RUN: would ${action} RunPod Pod ${pod_id:-<missing>}"
  exit 0
fi
if [[ -z "${pod_id}" ]]; then
  echo "RUNPOD_POD_ID is absent; no Pod lifecycle request was sent" >&2
  exit 1
fi

modern_cli_action() {
  command -v runpodctl >/dev/null 2>&1 || return 1
  runpodctl pod --help >/dev/null 2>&1 || return 1
  case "$1" in
    stop) runpodctl pod stop "${pod_id}" ;;
    delete) runpodctl pod delete "${pod_id}" ;;
  esac
}

legacy_cli_action() {
  command -v runpodctl >/dev/null 2>&1 || return 1
  case "$1" in
    stop) runpodctl stop pod "${pod_id}" ;;
    delete) runpodctl remove pod "${pod_id}" ;;
  esac
}

rest_action() {
  [[ -n "${api_key}" ]] || return 1
  command -v curl >/dev/null 2>&1 || return 1
  case "$1" in
    stop)
      curl --fail --show-error --silent \
        --request POST \
        --url "https://rest.runpod.io/v1/pods/${pod_id}/stop" \
        --header "Authorization: Bearer ${api_key}"
      ;;
    delete)
      curl --fail --show-error --silent \
        --request DELETE \
        --url "https://rest.runpod.io/v1/pods/${pod_id}" \
        --header "Authorization: Bearer ${api_key}"
      ;;
  esac
}

perform_action() {
  modern_cli_action "$1" && return 0
  legacy_cli_action "$1" && return 0
  rest_action "$1" && return 0
  return 1
}

if perform_action "${action}"; then
  exit 0
fi

# RunPod does not allow stopping a Pod with a network volume attached.  For
# this experiment RUNPOD_RESULTS_DURABLE=1 explicitly asserts that /workspace
# is durable, so deletion is the safe budget fuse if a requested stop fails.
if [[ "${action}" == "stop" && "${durable}" == "1" ]]; then
  echo "Pod stop failed; durable storage is declared, trying Pod deletion instead" >&2
  if perform_action delete; then
    exit 0
  fi
fi

echo "Unable to ${action} RunPod Pod ${pod_id} automatically" >&2
exit 1
