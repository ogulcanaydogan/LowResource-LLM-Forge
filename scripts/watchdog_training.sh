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
RECOVERY_REQUEST_FILE="${RECOVERY_REQUEST_FILE:-artifacts/logs/nan_recovery_request.env}"
MAX_RECOVERY_ATTEMPTS="${MAX_RECOVERY_ATTEMPTS:-5}"
LR_FLOOR="${LR_FLOOR:-1e-7}"
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
RECOVERY_REQUEST_FILE_ABS="$(abs_path "$RECOVERY_REQUEST_FILE")"
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
recovery_attempts=0
last_healthy_step=0

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

terminal_complete_event=0
if [[ -f "$LOG_FILE" ]] && tail -n +"$segment_start_line" "$LOG_FILE" | grep -a -q "training_complete"; then
    terminal_complete_event=1
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

# Check for structured recovery request from NaNGuardCallback (Level 1 → Level 2)
recovery_requested="no"
recovery_lr=""
recovery_checkpoint=""
recovery_field=""
if [[ -f "$RECOVERY_REQUEST_FILE_ABS" ]]; then
    # shellcheck disable=SC1090
    source "$RECOVERY_REQUEST_FILE_ABS" || true
    if [[ "${RECOVERY_REQUESTED:-}" == "1" ]]; then
        recovery_requested="yes"
        recovery_lr="${CURRENT_LR:-}"
        recovery_checkpoint="${LAST_CHECKPOINT:-}"
        recovery_field="${NAN_FIELD:-unknown}"
    fi
fi

# Fallback: also check log lines for NaN (catches cases where callback didn't write file)
latest_bad_line=""
if [[ -f "$LOG_FILE" ]]; then
    latest_bad_line="$(tail -n +"$segment_start_line" "$LOG_FILE" | grep -a -E -i "'(loss|grad_norm|eval_loss|entropy)':[[:space:]]*'?(nan|inf)'?|nan_guard_detected|nan_guard_stopping_training" | tail -n 1 || true)"
fi

new_nan_event="no"
if [[ "$recovery_requested" == "yes" ]]; then
    new_nan_event="yes"
elif [[ -n "$latest_bad_line" ]]; then
    signature="$(printf '%s' "$latest_bad_line" | sha256sum | awk '{print $1}')"
    if [[ "$signature" != "$last_nan_signature" ]]; then
        last_nan_signature="$signature"
        nan_hits=$((nan_hits + 1))
        new_nan_event="yes"
    fi
fi

halve_lr() {
    local lr="$1"
    awk "BEGIN { printf \"%.2e\", $lr / 2.0 }"
}

lr_below_floor() {
    local lr="$1"
    local floor="$2"
    awk "BEGIN { exit ($lr < $floor) ? 0 : 1 }" 2>/dev/null
}

clean_optimizer_state() {
    local ckpt_dir="$1"
    if [[ -d "$ckpt_dir" ]]; then
        rm -f "$ckpt_dir/optimizer.pt" "$ckpt_dir/optimizer.bin"
    fi
}

stop_training() {
    systemctl --user stop "$TRAINING_SERVICE_NAME" >/dev/null 2>&1 || true
    pkill -f "scripts/run_training.py --config" >/dev/null 2>&1 || true
    sleep 2
}

start_service_with_config() {
    local cfg="$1"
    local enable_resume="$2"
    local lr_override="${3:-}"
    local resume_from_override="${4:-}"
    local run_id

    upsert_env "TRAIN_CONFIG" "$cfg" "$FORGE_ENV_FILE_ABS"
    upsert_env "ENABLE_RESUME" "$enable_resume" "$FORGE_ENV_FILE_ABS"
    upsert_env "LR_OVERRIDE" "$lr_override" "$FORGE_ENV_FILE_ABS"

    if [[ -n "$resume_from_override" ]]; then
        upsert_env "RESUME_FROM" "$resume_from_override" "$FORGE_ENV_FILE_ABS"
    else
        # Policy restarts must use the run directory's latest safe checkpoint,
        # not a stale manual override from an earlier recovery.
        upsert_env "RESUME_FROM" "" "$FORGE_ENV_FILE_ABS"
    fi

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
    if [[ "$step" -ge "$TARGET_STEPS" ]] || [[ "$terminal_complete_event" -eq 1 ]]; then
        if [[ "$step" -ge "$TARGET_STEPS" ]]; then
            action="completed"
            message="target_step_reached"
        else
            action="completed_terminal_event"
            message="training_complete_seen_in_segment"
        fi
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
        # Progressive self-healing: halve LR and restart from checkpoint
        if [[ "$recovery_attempts" -ge "$MAX_RECOVERY_ATTEMPTS" ]]; then
            stop_training
            fatal_stopped=1
            action="fatal_stop_max_recovery"
            message="max recovery attempts ($MAX_RECOVERY_ATTEMPTS) exhausted"
            send_notify "LLM v100 stopped" "NaN recovery failed after $MAX_RECOVERY_ATTEMPTS attempts. Fatal lock." "rotating_light,warning"
        else
            # Determine current LR to halve
            current_lr="$recovery_lr"
            if [[ -z "$current_lr" ]] || [[ "$current_lr" == "0" ]] || [[ "$current_lr" == "0.0" ]]; then
                # No LR from recovery file; read from trainer_state.json in latest checkpoint
                current_lr=""
                if [[ -n "$recovery_checkpoint" ]] && [[ -f "$recovery_checkpoint/trainer_state.json" ]]; then
                    current_lr="$(python3 -c "import json; s=json.load(open('$recovery_checkpoint/trainer_state.json')); h=[e for e in s.get('log_history',[]) if 'learning_rate' in e]; print(h[-1]['learning_rate'] if h else '')" 2>/dev/null || true)"
                fi
            fi
            # Final fallback: use config default
            if [[ -z "$current_lr" ]] || [[ "$current_lr" == "0" ]] || [[ "$current_lr" == "0.0" ]]; then
                current_lr="$(python3 -c "
import yaml
with open('$PROJECT_ROOT/$TRAIN_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('training',{}).get('learning_rate', 1e-5))
" 2>/dev/null || echo "1e-5")"
            fi

            new_lr="$(halve_lr "$current_lr")"

            if lr_below_floor "$new_lr" "$LR_FLOOR"; then
                stop_training
                fatal_stopped=1
                action="fatal_stop_lr_floor"
                message="halved LR $new_lr below floor $LR_FLOOR"
                send_notify "LLM v100 stopped" "LR halved to $new_lr (below floor $LR_FLOOR). Fatal lock." "rotating_light,warning"
            else
                stop_training

                # Clean optimizer state from checkpoint to prevent moment mismatch
                ckpt_to_resume="$recovery_checkpoint"
                if [[ -z "$ckpt_to_resume" ]]; then
                    # Find latest checkpoint from training run dir
                    local_train_run_dir=""
                    case "$TRAIN_CONFIG" in
                        *turkcell_7b_v100_v3b_fallback.yaml)
                            local_train_run_dir="$PROJECT_ROOT/artifacts/training/turkcell-7b-sft-v100-v3b-fallback"
                            ;;
                        *turkcell_7b_v100_v3_ultrastable.yaml)
                            local_train_run_dir="$PROJECT_ROOT/artifacts/training/turkcell-7b-sft-v100-v3-ultrastable"
                            ;;
                        *)
                            local_train_run_dir="$PROJECT_ROOT/artifacts/training"
                            ;;
                    esac
                    if [[ -d "$local_train_run_dir" ]]; then
                        ckpt_to_resume="$(find "$local_train_run_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
                    fi
                fi

                if [[ -n "$ckpt_to_resume" ]]; then
                    clean_optimizer_state "$ckpt_to_resume"
                fi

                recovery_attempts=$((recovery_attempts + 1))
                nan_hits=0

                start_service_with_config "$TRAIN_CONFIG" "1" "$new_lr" "$ckpt_to_resume"

                action="recovery_restart"
                message="NaN on ${recovery_field:-unknown}; LR halved $current_lr -> $new_lr (attempt $recovery_attempts/$MAX_RECOVERY_ATTEMPTS)"
                send_notify "LLM v100 self-healing" "NaN detected. LR halved to $new_lr, restarting from checkpoint (attempt $recovery_attempts/$MAX_RECOVERY_ATTEMPTS)." "warning,repeat"
                last_step_ts="$now_epoch"
            fi
        fi

        # Clean up recovery request file after processing
        rm -f "$RECOVERY_REQUEST_FILE_ABS"
    else
        idle_seconds="$((now_epoch - last_step_ts))"
        restart_resume_flag="0"
        if [[ "$resume_armed" -eq 1 ]]; then
            restart_resume_flag="1"
        fi
        if [[ "$running" == "yes" ]] && [[ "$idle_seconds" -ge "$MAX_IDLE_SECONDS" ]] && [[ "$log_age" -ge "$MAX_LOG_STALE_SECONDS" ]]; then
            stop_training
            if [[ "$active_profile" == "fallback" ]]; then
                start_service_with_config "$FALLBACK_CONFIG" "$restart_resume_flag"
            else
                start_service_with_config "$TRAIN_CONFIG" "$restart_resume_flag"
            fi
            action="restart_stalled"
            message="stalled run restarted"
            send_notify "LLM v100 stalled restart" "No progress for ${idle_seconds}s; service restarted." "warning,repeat"
            last_step_ts="$now_epoch"
        elif [[ "$running" == "no" ]] && [[ "$step" -lt "$TARGET_STEPS" ]]; then
            if [[ "$active_profile" == "fallback" ]]; then
                start_service_with_config "$FALLBACK_CONFIG" "$restart_resume_flag"
            else
                start_service_with_config "$TRAIN_CONFIG" "$restart_resume_flag"
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

# Reset recovery state when training makes sustained progress after a recovery
if [[ "$recovery_attempts" -gt 0 ]] && [[ "$step" -gt "$last_healthy_step" ]]; then
    progress_since_recovery=$((step - last_healthy_step))
    # Consider training healthy after 100+ steps of progress without NaN
    if [[ "$progress_since_recovery" -ge 100 ]]; then
        recovery_attempts=0
        last_healthy_step="$step"
        upsert_env "LR_OVERRIDE" "" "$FORGE_ENV_FILE_ABS"
    fi
fi
if [[ "$step" -gt "$last_healthy_step" ]] && [[ "$new_nan_event" != "yes" ]]; then
    last_healthy_step="$step"
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
    echo "terminal_complete_event=$terminal_complete_event"
    echo "log_age_seconds=$log_age"
    echo "recovery_attempts=$recovery_attempts"
    echo "max_recovery_attempts=$MAX_RECOVERY_ATTEMPTS"
    echo "last_healthy_step=$last_healthy_step"
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
    echo "recovery_attempts=$recovery_attempts"
    echo "last_healthy_step=$last_healthy_step"
} > "$STATE_FILE_ABS"
