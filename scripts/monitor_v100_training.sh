#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
FORGE_ENV_FILE="${FORGE_ENV_FILE:-$HOME/.config/forge/v100_training.env}"
ACTIVE_RUN_FILE="${ACTIVE_RUN_FILE:-$PROJECT_ROOT/artifacts/logs/v100_active_run.env}"
STATUS_FILE="${STATUS_FILE:-$PROJECT_ROOT/artifacts/logs/training_monitor_status.txt}"

if [[ -f "$FORGE_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$FORGE_ENV_FILE"
fi

if [[ -f "$ACTIVE_RUN_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ACTIVE_RUN_FILE"
fi

cd "$PROJECT_ROOT"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/models/turkcell_7b_v100_v3_ultrastable.yaml}"
TARGET_STEPS="${TARGET_STEPS:-8601}"
TRAIN_LOG="${TRAIN_LOG:-artifacts/logs/training_full.log}"
RUN_ID="${RUN_ID:-unknown}"

abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$PROJECT_ROOT/$path"
    fi
}

LOG_FILE="$(abs_path "$TRAIN_LOG")"

running="no"
if pgrep -f "run_training.py --config ${TRAIN_CONFIG}" >/dev/null 2>&1 || pgrep -f "scripts/run_training.py" >/dev/null 2>&1; then
    running="yes"
fi

log_start_line=1
if [[ -f "$LOG_FILE" ]]; then
    if [[ "$RUN_ID" != "unknown" ]]; then
        marker_line="$(grep -a -n "forge-training-start run_id=${RUN_ID}" "$LOG_FILE" | tail -n 1 | cut -d ':' -f 1 || true)"
    else
        marker_line="$(grep -a -n "forge-training-start" "$LOG_FILE" | tail -n 1 | cut -d ':' -f 1 || true)"
    fi
    if [[ "$marker_line" =~ ^[0-9]+$ ]] && [[ "$marker_line" -gt 0 ]]; then
        log_start_line=$((marker_line + 1))
    fi
fi

progress="none"
if [[ -f "$LOG_FILE" ]]; then
    progress="$(tail -n +"$log_start_line" "$LOG_FILE" | grep -a -oE "[0-9]+/${TARGET_STEPS}" | tail -n 1 || true)"
    [[ -z "$progress" ]] && progress="none"
fi

step=0
if [[ "$progress" != "none" ]]; then
    step="${progress%%/*}"
fi

percent=0
if [[ "$step" =~ ^[0-9]+$ ]] && [[ "$TARGET_STEPS" -gt 0 ]]; then
    percent=$((step * 100 / TARGET_STEPS))
fi

nan_hits=0
if [[ -f "$LOG_FILE" ]]; then
    metric_nan_count="$(tail -n +"$log_start_line" "$LOG_FILE" | grep -a -E -i -c "'(loss|grad_norm|eval_loss|entropy)':[[:space:]]*'?(nan|inf)'?" || true)"
    guard_nan_count="$(tail -n +"$log_start_line" "$LOG_FILE" | grep -a -E -c "nan_guard_detected|nan_guard_stopping_training" || true)"
    metric_nan_count="${metric_nan_count:-0}"
    guard_nan_count="${guard_nan_count:-0}"
    nan_hits=$((metric_nan_count + guard_nan_count))
fi

gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
if [[ -z "$gpu_line" ]]; then
    gpu_line="unknown"
fi

state="stopped"
if [[ "$running" == "yes" ]]; then
    state="running"
fi
if [[ "$step" =~ ^[0-9]+$ ]] && [[ "$step" -ge "$TARGET_STEPS" ]]; then
    state="completed"
fi

{
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "running=$running"
    echo "step=$step"
    echo "target_steps=$TARGET_STEPS"
    echo "progress=$progress"
    echo "percent=$percent"
    echo "nan_hits=$nan_hits"
    echo "run_segment_id=$RUN_ID"
    echo "train_config=$TRAIN_CONFIG"
    echo "log_file=$TRAIN_LOG"
    echo "gpu=$gpu_line"
    echo "state=$state"
} >"$STATUS_FILE"
