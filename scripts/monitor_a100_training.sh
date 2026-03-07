#!/usr/bin/env bash
set -euo pipefail

cd /home/weezboo/projects/LowResource-LLM-Forge

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/models/turkcell_7b_a100_v8_stable_reset.yaml}"
CONFIG_BASENAME="$(basename "$TRAIN_CONFIG")"
CONFIG_SLUG="${CONFIG_BASENAME%.*}"
LOG_FILE="${LOG_FILE:-${TRAIN_LOG:-artifacts/logs/training_${CONFIG_SLUG}.log}}"
STATUS_FILE="${STATUS_FILE:-artifacts/logs/training_monitor_status_a100.txt}"
ETA_STATE_FILE="${ETA_STATE_FILE:-artifacts/logs/training_monitor_eta_state_${CONFIG_SLUG}.env}"
TARGET_STEPS="${TARGET_STEPS:-8601}"
PATTERN="${PATTERN:-run_training.py --config ${TRAIN_CONFIG}}"
SLEEP_SECS="${SLEEP_SECS:-60}"
SAVE_STEPS="${SAVE_STEPS:-250}"
CHECKPOINT_EVENT_FILE="${CHECKPOINT_EVENT_FILE:-artifacts/logs/training_checkpoint_events_${CONFIG_SLUG}.log}"
CHECKPOINT_STATE_FILE="${CHECKPOINT_STATE_FILE:-artifacts/logs/training_checkpoint_state_${CONFIG_SLUG}.env}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-}"

mkdir -p artifacts/logs

prev_ts=0
prev_step=0
ema_sps=""
speed_source="none"
last_announced_checkpoint_step=0

if [[ -z "$TRAIN_RUN_DIR" ]] && [[ -f "$LOG_FILE" ]]; then
    run_dir_from_log="$(grep -a -oE "output_dir=artifacts/training/[^[:space:]]+" "$LOG_FILE" | tail -n 1 | cut -d '=' -f 2 || true)"
    if [[ -n "$run_dir_from_log" ]]; then
        TRAIN_RUN_DIR="$run_dir_from_log"
    fi
fi

if [[ -z "$TRAIN_RUN_DIR" ]]; then
    TRAIN_RUN_DIR="artifacts/training/${CONFIG_SLUG}"
fi

if [[ -f "$ETA_STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ETA_STATE_FILE"
fi

if [[ -f "$CHECKPOINT_STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CHECKPOINT_STATE_FILE"
fi

if [[ ! "$last_announced_checkpoint_step" =~ ^[0-9]+$ ]]; then
    last_announced_checkpoint_step=0
fi

find_latest_checkpoint_step() {
    local current_step="$1"
    if [[ ! -d "$TRAIN_RUN_DIR" ]]; then
        echo "0"
        return
    fi

    latest="$(find "$TRAIN_RUN_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null \
        | sed -E 's#.*/checkpoint-##' \
        | grep -E '^[0-9]+$' \
        | awk -v s="$current_step" 's == "" || s !~ /^[0-9]+$/ || $1 <= s' \
        | sort -n \
        | tail -n 1 || true)"

    if [[ -z "$latest" ]]; then
        echo "0"
    else
        echo "$latest"
    fi
}

while true; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    now_epoch="$(date -u +%s)"

    running="no"
    if pgrep -f "$PATTERN" >/dev/null 2>&1 || pgrep -f "scripts/run_training.py" >/dev/null 2>&1; then
        running="yes"
    fi

    progress="none"
    log_start_line=1
    if [[ -f "$LOG_FILE" ]]; then
        marker_line="$(grep -a -n "forge-training-start" "$LOG_FILE" | tail -n 1 | cut -d ':' -f 1 || true)"
        if [[ "$marker_line" =~ ^[0-9]+$ ]] && [[ "$marker_line" -gt 0 ]]; then
            log_start_line=$((marker_line + 1))
        fi

        progress="$(tail -n +"$log_start_line" "$LOG_FILE" | grep -a -oE "[0-9]+/${TARGET_STEPS}" | tail -n 1 || true)"
        if [[ -z "$progress" ]]; then
            progress="none"
        fi
    fi

    step="0"
    if [[ "$progress" != "none" ]]; then
        step="${progress%%/*}"
    fi

    pct="0"
    if [[ "$step" =~ ^[0-9]+$ ]] && [[ $TARGET_STEPS -gt 0 ]]; then
        pct=$((step * 100 / TARGET_STEPS))
    fi

    nan_count="0"
    if [[ -f "$LOG_FILE" ]]; then
        # Count only real NaN/Inf metric values and explicit NaN guard events.
        metric_nan_count="$(tail -n +"$log_start_line" "$LOG_FILE" 2>/dev/null | grep -a -E -i "'(loss|grad_norm|eval_loss)':[[:space:]]*'?(nan|inf)'?" | wc -l | tr -d '[:space:]' || true)"
        guard_nan_count="$(tail -n +"$log_start_line" "$LOG_FILE" 2>/dev/null | grep -a -E -c "nan_guard_detected|nan_guard_stopping_training" || true)"
        metric_nan_count="${metric_nan_count:-0}"
        guard_nan_count="${guard_nan_count:-0}"
        nan_count=$((metric_nan_count + guard_nan_count))
    fi

    gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | head -n1 2>/dev/null || echo unknown)"

    steps_per_hour="unknown"
    eta_seconds="unknown"
    eta_utc="unknown"
    remaining_steps="unknown"
    latest_checkpoint_step="$(find_latest_checkpoint_step "$step")"
    next_checkpoint_step="unknown"
    steps_to_next_checkpoint="unknown"
    checkpoint_eta_utc="unknown"

    if [[ "$step" =~ ^[0-9]+$ ]] && [[ "$step" -lt "$TARGET_STEPS" ]]; then
        remaining_steps=$((TARGET_STEPS - step))
    fi

    if [[ "$step" =~ ^[0-9]+$ ]] && [[ "$SAVE_STEPS" =~ ^[0-9]+$ ]] && [[ "$SAVE_STEPS" -gt 0 ]]; then
        next_checkpoint_step=$((((step / SAVE_STEPS) + 1) * SAVE_STEPS))
        if [[ "$next_checkpoint_step" -le "$TARGET_STEPS" ]]; then
            steps_to_next_checkpoint=$((next_checkpoint_step - step))
        else
            next_checkpoint_step="none"
            steps_to_next_checkpoint="none"
        fi
    fi

    if [[ "$step" =~ ^[0-9]+$ ]] && [[ "$prev_ts" =~ ^[0-9]+$ ]] && [[ "$prev_step" =~ ^[0-9]+$ ]]; then
        if [[ $prev_ts -gt 0 ]] && [[ $now_epoch -gt $prev_ts ]] && [[ $step -gt $prev_step ]]; then
            delta_steps=$((step - prev_step))
            delta_secs=$((now_epoch - prev_ts))
            # Skip the first large jump after resume (e.g. 0 -> 750) to avoid bogus ETA.
            if [[ -z "$ema_sps" ]] && [[ "$prev_step" -eq 0 ]] && [[ "$delta_steps" -gt 1 ]]; then
                speed_source="bootstrap_skip"
            elif [[ "$delta_secs" -gt 0 ]]; then
                instant_sps="$(awk -v ds="$delta_steps" -v dt="$delta_secs" 'BEGIN { printf "%.8f", ds / dt }')"

                if [[ -n "$ema_sps" ]]; then
                    ema_sps="$(awk -v e="$ema_sps" -v i="$instant_sps" 'BEGIN { printf "%.8f", (0.7 * e) + (0.3 * i) }')"
                    speed_source="ema"
                else
                    ema_sps="$instant_sps"
                    speed_source="instant"
                fi
            fi
        fi
    fi

    if [[ -n "$ema_sps" ]] && awk -v s="$ema_sps" 'BEGIN { exit !(s > 0) }'; then
        steps_per_hour="$(awk -v s="$ema_sps" 'BEGIN { printf "%.1f", s * 3600 }')"
        if [[ "$remaining_steps" =~ ^[0-9]+$ ]]; then
            eta_seconds="$(awk -v rem="$remaining_steps" -v s="$ema_sps" 'BEGIN { printf "%.0f", rem / s }')"
            if [[ "$eta_seconds" =~ ^[0-9]+$ ]]; then
                eta_utc="$(date -u -d "@$((now_epoch + eta_seconds))" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
            fi
        fi
        if [[ "$steps_to_next_checkpoint" =~ ^[0-9]+$ ]]; then
            checkpoint_eta_seconds="$(awk -v rem="$steps_to_next_checkpoint" -v s="$ema_sps" 'BEGIN { printf "%.0f", rem / s }')"
            if [[ "$checkpoint_eta_seconds" =~ ^[0-9]+$ ]]; then
                checkpoint_eta_utc="$(date -u -d "@$((now_epoch + checkpoint_eta_seconds))" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
            fi
        fi
    fi

    if [[ "$latest_checkpoint_step" =~ ^[0-9]+$ ]] && [[ "$latest_checkpoint_step" -gt "$last_announced_checkpoint_step" ]]; then
        echo "${ts} checkpoint_saved step=${latest_checkpoint_step} run_dir=${TRAIN_RUN_DIR}" >>"$CHECKPOINT_EVENT_FILE"
        last_announced_checkpoint_step="$latest_checkpoint_step"
    fi

    {
        echo "timestamp_utc=$ts"
        echo "running=$running"
        echo "step=$step"
        echo "target_steps=$TARGET_STEPS"
        echo "progress=$progress"
        echo "percent=$pct"
        echo "remaining_steps=$remaining_steps"
        echo "steps_per_hour=$steps_per_hour"
        echo "eta_seconds=$eta_seconds"
        echo "eta_utc=$eta_utc"
        echo "speed_source=$speed_source"
        echo "nan_count=$nan_count"
        echo "gpu=$gpu_line"
        echo "save_steps=$SAVE_STEPS"
        echo "latest_checkpoint_step=$latest_checkpoint_step"
        echo "next_checkpoint_step=$next_checkpoint_step"
        echo "steps_to_next_checkpoint=$steps_to_next_checkpoint"
        echo "checkpoint_eta_utc=$checkpoint_eta_utc"
    } >"$STATUS_FILE"

    prev_ts="$now_epoch"
    prev_step="$step"
    {
        echo "prev_ts=$prev_ts"
        echo "prev_step=$prev_step"
        echo "ema_sps=$ema_sps"
        echo "speed_source=$speed_source"
    } >"$ETA_STATE_FILE"
    {
        echo "last_announced_checkpoint_step=$last_announced_checkpoint_step"
        echo "train_run_dir=$TRAIN_RUN_DIR"
    } >"$CHECKPOINT_STATE_FILE"

    if [[ "$running" == "no" ]]; then
        echo "state=stopped" >>"$STATUS_FILE"
        exit 0
    fi

    if [[ "$step" =~ ^[0-9]+$ ]] && [[ $step -ge $TARGET_STEPS ]]; then
        echo "state=completed" >>"$STATUS_FILE"
        exit 0
    fi

    echo "state=running" >>"$STATUS_FILE"
    sleep "$SLEEP_SECS"
done
