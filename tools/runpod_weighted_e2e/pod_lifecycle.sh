#!/usr/bin/env bash
set -euo pipefail

action="${1:?usage: pod_lifecycle.sh <stop|delete>}"
pod_id="${RUNPOD_POD_ID:-}"

if [[ "${RUNPOD_DRY_RUN:-0}" == "1" ]]; then
  echo "DRY RUN: would ${action} RunPod Pod ${pod_id:-<missing>}"
  exit 0
fi
if [[ -z "${pod_id}" ]]; then
  echo "RUNPOD_POD_ID is absent; no Pod lifecycle request was sent" >&2
  exit 1
fi

if command -v runpodctl >/dev/null 2>&1; then
  case "${action}" in
    stop) runpodctl pod stop "${pod_id}" ;;
    delete) runpodctl pod delete "${pod_id}" ;;
    *) echo "unsupported Pod action: ${action}" >&2; exit 2 ;;
  esac
  exit 0
fi

api_key="${RUNPOD_API_KEY:-}"
if [[ -z "${api_key}" ]]; then
  echo "Neither runpodctl nor RUNPOD_API_KEY is available" >&2
  exit 1
fi
case "${action}" in
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
  *)
    echo "unsupported Pod action: ${action}" >&2
    exit 2
    ;;
esac
