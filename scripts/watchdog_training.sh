#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
FORGE_ENV_FILE="${FORGE_ENV_FILE:-$HOME/.config/forge/v100_training.env}"
ACTIVE_RUN_FILE="${ACTIVE_RUN_FILE:-$PROJECT_ROOT/artifacts/logs/v100_active_run.env}"
STATUS_FILE="${STATUS_FILE:-$PROJECT_ROOT/artifacts/logs/training_watchdog_status.txt}"
STATE_FILE="${STATE_FILE:-$PROJECT_ROOT/artifacts/logs/training_watchdog_state.env}"
MONITOR_STATUS_FILE="${MONITOR_STATUS_FILE:-$PROJECT_ROOT/artifacts/logs/training_monitor_status.txt}"
TRAINING_SERVICE_NAME="${TRAINING_SERVICE_NAME:-forge-v100-training.service}"
COMPLETION_SCRIPT="${COMPLETION_SCRIPT:-$PROJECT_ROOT/scripts/run_v100_completion.sh}"

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
FALLBACK_CONFIG="${FALLBACK_CONFIG:-configs/models/turkcell_7b_v100_v3b_fallback.yaml}"
TARGET_STEPS="${TARGET_STEPS:-8601}"
MAX_IDLE_SECONDS="${MAX_IDLE_SECONDS:-5400}"
MAX_LOG_STALE_SECONDS="${MAX_LOG_STALE_SECONDS:-900}"
RESUME_AFTER_STEP="${RESUME_AFTER_STEP:-500}"
RUN_ID="${RUN_ID:-unknown}"
TRAIN_LOG="${TRAIN_LOG:-artifacts/logs/training_full.log}"
TOPIC="${FORGE_NOTIFY_TOPIC:-weezboo-forge-training}"
NTFY_URL="${FORGE_NOTIFY_URL:-https://ntfy.sh/${TOPIC}}"

abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$PROJECT_ROOT/$path"
    fi
}

upsert_env() {
    local key="$1"
    local value="$2"
    local file="$3"
    mkdir -p "$(dirname "$file")"
    if [[ -f "$file" ]] && grep -q "^${key}=" "$file"; then
        sed -i "s#^${key}=.*#${key}=${value}#" "$file"
    else
        echo "${key}=${value}" >>"$file"
    fi
}

send_notify() {
    local title="$1"
    local body="$2"
    local tags="${3:-warning}"
    curl -fsS -m 20 \
        -H "Title: ${title}" \
        -H "Tags: ${tags}" \
        -H "Priority: high" \
        -d "$body" \
        "$NTFY_URL" >/dev/null || true
}

LOG_FILE="$(abs_path "$TRAIN_LOG")"
STATUS_FILE_ABS="$(abs_path "$STATUS_FILE")"
STATE_FILE_ABS="$(abs_path "$STATE_FILE")"
FORGE_ENV_FILE_ABS="$(abs_path "$FORGE_ENV_FILE")"
MONITOR_STATUS_FILE_ABS="$(abs_path "$MONITOR_STATUS_FILE")"
mkdir -p "$(dirname "$STATUS_FILE_ABS")"

fallback_applied=0
fatal_stopped=0
last_step=0
last_step_ts=0
nan_hits=0
resume_armed=0
active_profile="primary"
last_nan_signature=""
posttrain_triggered=0

if [[ -f "$STATE_FILE_ABS" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE_ABS" || true
fi

segment_start_line=1
if [[ -f "$LOG_FILE" ]]; then
    if [[ "$RUN_ID" != "unknown" ]]; then
        marker_line="$(grep -a -n "forge-training-start run_id=${RUN_ID}" "$LOG_FILE" | tail -n 1 | cut -d ':' -f 1 || true)"
    else
        marker_line="$(grep -a -n "forge-training-start" "$LOG_FILE" | tail -n 1 | cut -d ':' -f 1 || true)"
    fi
    if [[ "$marker_line" =~ ^[0-9]+$ ]] && [[ "$marker_line" -gt 0 ]]; then
        segment_start_line=$((marker_line + 1))
    fi
fi

progress="none"
if [[ -f "$LOG_FILE" ]]; then
    progress="$(tail -n +"$segment_start_line" "$LOG_FILE" | grep -a -oE "[0-9]+/${TARGET_STEPS}" | tail -n 1 || true)"
    [[ -z "$progress" ]] && progress="none"
fi

step=0
if [[ "$progress" != "none" ]]; then
    step="${progress%%/*}"
fi

service_state="$(systemctl --user is-active "$TRAINING_SERVICE_NAME" 2>/dev/null || true)"
running="no"
if [[ "$service_state" == "active" ]] || [[ "$service_state" == "activating" ]]; then
    running="yes"
fi

now_epoch="$(date +%s)"
if [[ -f "$LOG_FILE" ]]; then
    log_mtime_epoch="$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0)"
else
    log_mtime_epoch=0
fi
log_age="$((now_epoch - log_mtime_epoch))"

if [[ "$step" -gt "$last_step" ]]; then
    last_step="$step"
    last_step_ts="$now_epoch"
elif [[ "$last_step_ts" -eq 0 ]]; then
    last_step_ts="$now_epoch"
fi

latest_bad_line=""
if [[ -f "$LOG_FILE" ]]; then
    latest_bad_line="$(tail -n +"$segment_start_line" "$LOG_FILE" | grep -a -E -i "'(loss|grad_norm|eval_loss|entropy)':[[:space:]]*'?(nan|inf)'?|nan_guard_detected|nan_guard_stopping_training" | tail -n 1 || true)"
fi

new_nan_event="no"
if [[ -n "$latest_bad_line" ]]; then
    signature="$(printf '%s' "$latest_bad_line" | sha256sum | awk '{print $1}')"
    if [[ "$signature" != "$last_nan_signature" ]]; then
        last_nan_signature="$signature"
        nan_hits=$((nan_hits + 1))
        new_nan_event="yes"
    fi
fi

stop_training() {
    systemctl --user stop "$TRAINING_SERVICE_NAME" >/dev/null 2>&1 || true
    pkill -f "scripts/run_training.py --config" >/dev/null 2>&1 || true
    sleep 2
}

start_service_with_config() {
    local cfg="$1"
    local enable_resume="$2"
    local run_id

    upsert_env "TRAIN_CONFIG" "$cfg" "$FORGE_ENV_FILE_ABS"
    upsert_env "ENABLE_RESUME" "$enable_resume" "$FORGE_ENV_FILE_ABS"

    run_id="$(date -u +%Y%m%dT%H%M%SZ)"
    systemctl --user set-environment RUN_ID="$run_id"
    systemctl --user start "$TRAINING_SERVICE_NAME"

    RUN_ID="$run_id"
    TRAIN_CONFIG="$cfg"
}

action="none"
message="ok"

if [[ "$fatal_stopped" -eq 1 ]]; then
    action="fatal_locked"
    message="watchdog_fatal_stop_active"
else
    if [[ "$step" -ge "$TARGET_STEPS" ]]; then
        action="completed"
        message="target_step_reached"
        if [[ "$posttrain_triggered" -eq 0 ]] && [[ -x "$COMPLETION_SCRIPT" ]]; then
            if bash "$COMPLETION_SCRIPT"; then
                posttrain_triggered=1
                action="posttrain_triggered"
                message="completion_pipeline_started"
            else
                action="posttrain_failed"
                message="completion_pipeline_failed"
                send_notify "LLM v100 completion failed" "Post-training pipeline failed after step completion." "warning,rotating_light"
            fi
        fi
    elif [[ "$new_nan_event" == "yes" ]]; then
        if [[ "$fallback_applied" -eq 0 ]] && [[ "$active_profile" != "fallback" ]]; then
            stop_training
            start_service_with_config "$FALLBACK_CONFIG" "0"
            fallback_applied=1
            active_profile="fallback"
            nan_hits=0
            action="fallback_restart"
            message="nan detected on primary; fallback started"
            send_notify "LLM v100 fallback restart" "NaN/Inf detected on primary profile. Restarted with fallback config." "warning,repeat"
            last_step_ts="$now_epoch"
        else
            stop_training
            fatal_stopped=1
            action="fatal_stop_after_fallback"
            message="nan detected on fallback; training stopped"
            send_notify "LLM v100 stopped" "NaN/Inf detected on fallback. Fatal lock enabled." "rotating_light,warning"
        fi
    else
        idle_seconds="$((now_epoch - last_step_ts))"
        if [[ "$running" == "yes" ]] && [[ "$idle_seconds" -ge "$MAX_IDLE_SECONDS" ]] && [[ "$log_age" -ge "$MAX_LOG_STALE_SECONDS" ]]; then
            stop_training
            if [[ "$active_profile" == "fallback" ]]; then
                start_service_with_config "$FALLBACK_CONFIG" "0"
            else
                start_service_with_config "$TRAIN_CONFIG" "0"
            fi
            action="restart_stalled"
            message="stalled run restarted"
            send_notify "LLM v100 stalled restart" "No progress for ${idle_seconds}s; service restarted." "warning,repeat"
            last_step_ts="$now_epoch"
        elif [[ "$running" == "no" ]] && [[ "$step" -lt "$TARGET_STEPS" ]]; then
            if [[ "$active_profile" == "fallback" ]]; then
                start_service_with_config "$FALLBACK_CONFIG" "0"
            else
                start_service_with_config "$TRAIN_CONFIG" "0"
            fi
            action="restart_down"
            message="service down; restarted"
            send_notify "LLM v100 restarted" "Training service was down and restarted." "warning,repeat"
            last_step_ts="$now_epoch"
        fi
    fi
fi

if [[ "$resume_armed" -eq 0 ]] && [[ "$step" -ge "$RESUME_AFTER_STEP" ]] && [[ "$nan_hits" -eq 0 ]]; then
    upsert_env "ENABLE_RESUME" "1" "$FORGE_ENV_FILE_ABS"
    resume_armed=1
fi

bash "$PROJECT_ROOT/scripts/monitor_v100_training.sh" || true

{
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "running=$running"
    echo "step=$step"
    echo "target_steps=$TARGET_STEPS"
    echo "run_segment_id=$RUN_ID"
    echo "active_profile=$active_profile"
    echo "nan_hits=$nan_hits"
    echo "fallback_applied=$fallback_applied"
    echo "fatal_stopped=$fatal_stopped"
    echo "resume_armed=$resume_armed"
    echo "posttrain_triggered=$posttrain_triggered"
    echo "log_age_seconds=$log_age"
    echo "action=$action"
    echo "message=$message"
} > "$STATUS_FILE_ABS"

{
    echo "fallback_applied=$fallback_applied"
    echo "fatal_stopped=$fatal_stopped"
    echo "last_step=$last_step"
    echo "last_step_ts=$last_step_ts"
    echo "nan_hits=$nan_hits"
    echo "resume_armed=$resume_armed"
    echo "active_profile=$active_profile"
    echo "last_nan_signature=$last_nan_signature"
    echo "posttrain_triggered=$posttrain_triggered"
} > "$STATE_FILE_ABS"
