#!/usr/bin/env bash
# run_notebook_tmux_papermill_git_terminate.sh
#
# Purpose:
#   1. Start/continue inside a tmux session.
#   2. Run a Jupyter notebook with Papermill.
#   3. git add / commit / push the results.
#   4. Terminate, stop, or do nothing to the current RunPod pod.
#
# Typical usage from inside /workspace/condition_df:
#
#   chmod +x run_notebook_tmux_papermill_git_terminate.sh
#
#   export RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY"   # do NOT commit this
#   ./run_notebook_tmux_papermill_git_terminate.sh \
#     notebooks/papermill_tmux_test.ipynb \
#     runs/papermill_test_output.ipynb \
#     "Run papermill test"
#
# Reattach while it is running:
#
#   tmux attach -t pm_run
#
# Safer test mode that does NOT stop/delete the pod:
#
#   RUNPOD_ACTION=none ./run_notebook_tmux_papermill_git_terminate.sh \
#     notebooks/papermill_tmux_test.ipynb \
#     runs/papermill_test_output.ipynb \
#     "Run papermill test"
#
# Important:
#   - RUNPOD_ACTION=delete permanently terminates/deletes the pod after success.
#   - The script does not store your RunPod API key. Provide it via env var.
#   - By default, it runs `git add -A`. Use GIT_ADD_PATHS to narrow this.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_notebook_tmux_papermill_git_terminate.sh INPUT_NOTEBOOK [OUTPUT_NOTEBOOK] [COMMIT_MESSAGE]

Example:
  export RUNPOD_API_KEY="..."
  ./run_notebook_tmux_papermill_git_terminate.sh \
    notebooks/experiment_8.ipynb \
    runs/experiment_8_output.ipynb \
    "Run experiment 8"

Useful environment variables:
  SESSION_NAME=pm_run                 tmux session name
  AUTOSAVE_SECONDS=60                 Papermill autosave interval
  KERNEL_NAME=condition_df_venv        Optional Jupyter kernel name
  EXTRA_PAPERMILL_ARGS="..."          Optional extra Papermill args
  GIT_ADD_PATHS="notebooks scripts"   Paths to git add; default: -A
  FORCE_ADD_OUTPUT=1                 Force-add output notebook and log even if ignored
  GIT_REMOTE=origin                   Git remote
  GIT_BRANCH=main                     Branch to push; default: current branch
  GIT_PULL_BEFORE_PUSH=0              Set 1 to run git pull --rebase --autostash first
  MAX_GIT_FILE_MB=95                  Refuse staged files bigger than this unless ALLOW_LARGE_GIT_FILES=1
  ALLOW_LARGE_GIT_FILES=0             Set 1 to allow large staged files
  RUNPOD_ACTION=delete                delete | stop | none
  TERMINATE_ON_FAILURE=0              Set 1 to stop/delete even if notebook/git fails
  RUNPOD_API_KEY=...                  Required for REST API stop/delete unless runpodctl is available
  RUNPOD_POD_ID=...                   Usually already provided by RunPod
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

INPUT_NOTEBOOK="$1"
OUTPUT_NOTEBOOK="${2:-runs/$(basename "${INPUT_NOTEBOOK%.ipynb}")_output_$(date +%Y%m%d_%H%M%S).ipynb}"
COMMIT_MESSAGE="${3:-Run notebook $(basename "$INPUT_NOTEBOOK") at $(date -u +%Y-%m-%dT%H:%M:%SZ)}"

SESSION_NAME="${SESSION_NAME:-pm_run}"
AUTOSAVE_SECONDS="${AUTOSAVE_SECONDS:-60}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
RUNPOD_ACTION="${RUNPOD_ACTION:-delete}"   # delete | stop | none
TERMINATE_ON_FAILURE="${TERMINATE_ON_FAILURE:-0}"
MAX_GIT_FILE_MB="${MAX_GIT_FILE_MB:-95}"
ALLOW_LARGE_GIT_FILES="${ALLOW_LARGE_GIT_FILES:-0}"
FORCE_ADD_OUTPUT="${FORCE_ADD_OUTPUT:-1}"
GIT_PULL_BEFORE_PUSH="${GIT_PULL_BEFORE_PUSH:-0}"

# Resolve paths before entering tmux so the same cwd is used.
START_DIR="$(pwd)"
SCRIPT_PATH="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$0")"

# If not already inside the tmux-controlled run, start detached tmux and exit.
if [[ "${INSIDE_TMUX_AUTORUN:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed. Install it first, e.g. apt-get update && apt-get install -y tmux"
    exit 1
  fi

  mkdir -p "$(dirname "$OUTPUT_NOTEBOOK")" logs

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION_NAME' already exists."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    echo "Or choose a new name: SESSION_NAME=my_new_run $0 ..."
    exit 1
  fi

  echo "Starting detached tmux session: $SESSION_NAME"
  echo "Notebook: $INPUT_NOTEBOOK"
  echo "Output notebook: $OUTPUT_NOTEBOOK"
  echo "Attach with: tmux attach -t $SESSION_NAME"

  q() { printf '%q' "$1"; }
  tmux_cmd="cd $(q "$START_DIR") && INSIDE_TMUX_AUTORUN=1 bash $(q "$SCRIPT_PATH") $(q "$INPUT_NOTEBOOK") $(q "$OUTPUT_NOTEBOOK") $(q "$COMMIT_MESSAGE")"
  tmux new-session -d -s "$SESSION_NAME" "$tmux_cmd"

  exit 0
fi

# From here onward, we are inside tmux.
mkdir -p "$(dirname "$OUTPUT_NOTEBOOK")" logs

LOG_FILE="${LOG_FILE:-logs/$(basename "${OUTPUT_NOTEBOOK%.ipynb}")_$(date +%Y%m%d_%H%M%S).log}"
touch "$LOG_FILE"

# Duplicate all script output to the log.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Started: $(date -Is)"
echo "Working directory: $(pwd)"
echo "Input notebook: $INPUT_NOTEBOOK"
echo "Output notebook: $OUTPUT_NOTEBOOK"
echo "Log file: $LOG_FILE"
echo "RunPod action after success: $RUNPOD_ACTION"
echo "============================================================"

finish_runpod_action() {
  local status="$1"

  if [[ "$status" -ne 0 && "$TERMINATE_ON_FAILURE" != "1" ]]; then
    echo "Run failed with status $status. Not stopping/deleting RunPod because TERMINATE_ON_FAILURE=0."
    echo "Inspect the run with: tmux attach -t $SESSION_NAME"
    return 0
  fi

  if [[ "$RUNPOD_ACTION" == "none" ]]; then
    echo "RUNPOD_ACTION=none, so leaving pod running."
    return 0
  fi

  local pod_id="${RUNPOD_POD_ID:-}"
  if [[ -z "$pod_id" ]]; then
    echo "ERROR: RUNPOD_POD_ID is not set, so I cannot $RUNPOD_ACTION the pod."
    echo "Set RUNPOD_POD_ID manually or use RUNPOD_ACTION=none."
    return 1
  fi

  if [[ "$RUNPOD_ACTION" == "stop" ]]; then
    echo "Stopping RunPod pod: $pod_id"
    if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
      curl --fail-with-body --silent --show-error \
        --request POST \
        --url "https://rest.runpod.io/v1/pods/${pod_id}/stop" \
        --header "Authorization: Bearer ${RUNPOD_API_KEY}"
      echo
      echo "Stop request sent."
    elif command -v runpodctl >/dev/null 2>&1; then
      runpodctl pod stop "$pod_id"
    else
      echo "ERROR: Need RUNPOD_API_KEY or runpodctl to stop the pod."
      return 1
    fi
  elif [[ "$RUNPOD_ACTION" == "delete" ]]; then
    echo "Terminating/deleting RunPod pod: $pod_id"
    echo "WARNING: This permanently deletes pod data not stored in a network volume."
    if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
      curl --fail-with-body --silent --show-error \
        --request DELETE \
        --url "https://rest.runpod.io/v1/pods/${pod_id}" \
        --header "Authorization: Bearer ${RUNPOD_API_KEY}"
      echo
      echo "Delete request sent. This session may disappear soon."
    elif command -v runpodctl >/dev/null 2>&1; then
      runpodctl pod delete "$pod_id"
    else
      echo "ERROR: Need RUNPOD_API_KEY or runpodctl to delete the pod."
      return 1
    fi
  else
    echo "ERROR: Unknown RUNPOD_ACTION='$RUNPOD_ACTION'. Use delete, stop, or none."
    return 1
  fi
}

on_exit() {
  local status="$?"
  if [[ "$status" -ne 0 ]]; then
    echo "============================================================"
    echo "FAILED with status $status at $(date -Is)"
    echo "============================================================"
    finish_runpod_action "$status" || true
  fi
}
trap on_exit EXIT

if [[ ! -f "$INPUT_NOTEBOOK" ]]; then
  echo "ERROR: Input notebook does not exist: $INPUT_NOTEBOOK"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed."
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: Current directory is not inside a Git repository."
  exit 1
fi

# Install papermill if needed.
if ! python -c "import papermill" >/dev/null 2>&1; then
  echo "Papermill not found in current Python environment. Installing papermill and ipykernel..."
  python -m pip install papermill ipykernel
fi

# Optional pull before run.
if [[ "$GIT_PULL_BEFORE_PUSH" == "1" ]]; then
  echo "Running git pull --rebase --autostash before notebook execution..."
  git pull --rebase --autostash "$GIT_REMOTE"
fi

# Build papermill command.
PAPERMILL_CMD=(python -m papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK"
  --log-output
  --request-save-on-cell-execute
  --autosave-cell-every "$AUTOSAVE_SECONDS"
)

if [[ -n "${KERNEL_NAME:-}" ]]; then
  PAPERMILL_CMD+=(-k "$KERNEL_NAME")
fi

echo "Running Papermill..."
printf 'Command: '
printf '%q ' "${PAPERMILL_CMD[@]}"
if [[ -n "${EXTRA_PAPERMILL_ARGS:-}" ]]; then
  printf '%s ' "$EXTRA_PAPERMILL_ARGS"
fi
echo

# shellcheck disable=SC2086
"${PAPERMILL_CMD[@]}" ${EXTRA_PAPERMILL_ARGS:-}

echo "Papermill finished successfully at $(date -Is)."

echo "Git status before adding:"
git status --short

if [[ -n "${GIT_ADD_PATHS:-}" ]]; then
  echo "Running: git add $GIT_ADD_PATHS"
  # shellcheck disable=SC2086
  git add $GIT_ADD_PATHS
else
  echo "Running: git add -A"
  git add -A
fi

if [[ "$FORCE_ADD_OUTPUT" == "1" ]]; then
  echo "Force-adding output notebook and log, in case runs/ or logs/ are ignored."
  git add -f "$OUTPUT_NOTEBOOK" "$LOG_FILE"
fi

# Refuse very large staged files unless explicitly allowed.
if [[ "$ALLOW_LARGE_GIT_FILES" != "1" ]]; then
  max_bytes=$((MAX_GIT_FILE_MB * 1024 * 1024))
  too_large=0
  while IFS= read -r -d '' f; do
    if [[ -f "$f" ]]; then
      size=$(wc -c < "$f" | tr -d ' ')
      if [[ "$size" -gt "$max_bytes" ]]; then
        size_mb="$(python -c 'import sys; print(f"{int(sys.argv[1])/1024/1024:.1f} MB")' "$size")"
        echo "ERROR: staged file is larger than ${MAX_GIT_FILE_MB}MB: $f ($size_mb)"
        too_large=1
      fi
    fi
  done < <(git diff --cached --name-only -z)

  if [[ "$too_large" == "1" ]]; then
    echo "Refusing to commit large files. Either remove them from staging or set ALLOW_LARGE_GIT_FILES=1."
    exit 1
  fi
fi

if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  echo "Committing changes..."
  git commit -m "$COMMIT_MESSAGE"
fi

BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
if [[ "$BRANCH" == "HEAD" || -z "$BRANCH" ]]; then
  echo "ERROR: Git is in detached HEAD state. Set GIT_BRANCH manually."
  exit 1
fi

echo "Pushing to $GIT_REMOTE $BRANCH..."
git push "$GIT_REMOTE" "$BRANCH"

echo "============================================================"
echo "SUCCESS at $(date -Is)"
echo "Output notebook: $OUTPUT_NOTEBOOK"
echo "Log file: $LOG_FILE"
echo "Git push succeeded."
echo "============================================================"

# Avoid the EXIT failure handler from running after successful RunPod action.
trap - EXIT
finish_runpod_action 0
