#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
FORGE_ENV_FILE="${FORGE_ENV_FILE:-$HOME/.config/forge/v100_training.env}"
PID_FILE="${PID_FILE:-$PROJECT_ROOT/artifacts/logs/training_full.pid}"
TRAINING_SERVICE_NAME="${TRAINING_SERVICE_NAME:-forge-v100-training.service}"
FORCE_RESTART="${FORCE_RESTART:-0}"

if [[ -f "$FORGE_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$FORGE_ENV_FILE"
fi

cd "$PROJECT_ROOT"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/models/turkcell_7b_v100_v3_ultrastable.yaml}"
RUN_PATTERN="run_training.py --config ${TRAIN_CONFIG}"

mkdir -p "$(dirname "$PID_FILE")"

if [[ "$FORCE_RESTART" != "1" ]] && systemctl --user is-active --quiet "$TRAINING_SERVICE_NAME"; then
    echo "training_service_active"
    systemctl --user --no-pager --lines=0 status "$TRAINING_SERVICE_NAME" | head -n 1 || true
    exit 0
fi

if [[ "$FORCE_RESTART" != "1" ]] && pgrep -f "$RUN_PATTERN" >/dev/null 2>&1; then
    echo "training_already_running"
    pgrep -af "$RUN_PATTERN" | head -n 4
    exit 0
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
systemctl --user set-environment RUN_ID="$RUN_ID"
if [[ "$FORCE_RESTART" == "1" ]]; then
    systemctl --user restart "$TRAINING_SERVICE_NAME"
else
    systemctl --user start "$TRAINING_SERVICE_NAME"
fi

sleep 2
service_state="$(systemctl --user is-active "$TRAINING_SERVICE_NAME" || true)"
if [[ "$service_state" != "active" ]] && [[ "$service_state" != "activating" ]]; then
    echo "training_service_failed"
    systemctl --user --no-pager --lines=50 status "$TRAINING_SERVICE_NAME" || true
    exit 1
fi

if pgrep -f "scripts/run_training.py" >/dev/null 2>&1; then
    pgrep -f "scripts/run_training.py" | head -n 1 > "$PID_FILE"
fi

echo "service=$TRAINING_SERVICE_NAME"
echo "service_state=$service_state"
echo "run_id=$RUN_ID"
echo "train_config=$TRAIN_CONFIG"
echo "pid_file=$PID_FILE"
