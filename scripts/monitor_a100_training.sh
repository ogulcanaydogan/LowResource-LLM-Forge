#!/usr/bin/env bash
set -euo pipefail

cd /home/weezboo/projects/LowResource-LLM-Forge

LOG_FILE="${LOG_FILE:-artifacts/logs/training_a100_bf16_v4_recovery.log}"
STATUS_FILE="${STATUS_FILE:-artifacts/logs/training_monitor_status_a100.txt}"
ETA_STATE_FILE="${ETA_STATE_FILE:-artifacts/logs/training_monitor_eta_state_a100.env}"
TARGET_STEPS="${TARGET_STEPS:-8601}"
PATTERN="${PATTERN:-run_training.py --config configs/models/turkcell_7b_a100_v4_recovery.yaml}"
SLEEP_SECS="${SLEEP_SECS:-60}"

mkdir -p artifacts/logs

prev_ts=0
prev_step=0
ema_sps=""
speed_source="none"

if [[ -f "$ETA_STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ETA_STATE_FILE"
fi

while true; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    now_epoch="$(date -u +%s)"

    running="no"
    if pgrep -f "$PATTERN" >/dev/null 2>&1; then
        running="yes"
    fi

    progress="none"
    if [[ -f "$LOG_FILE" ]]; then
        progress="$(grep -a -oE "[0-9]+/${TARGET_STEPS}" "$LOG_FILE" | tail -n 1 || true)"
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
        metric_nan_count="$(grep -a -E -i "'(loss|grad_norm|eval_loss)':[[:space:]]*'?(nan|inf)'?" "$LOG_FILE" | wc -l | tr -d '[:space:]' || true)"
        guard_nan_count="$(grep -a -E -c "nan_guard_detected|nan_guard_stopping_training" "$LOG_FILE" || true)"
        metric_nan_count="${metric_nan_count:-0}"
        guard_nan_count="${guard_nan_count:-0}"
        nan_count=$((metric_nan_count + guard_nan_count))
    fi

    gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | head -n1 2>/dev/null || echo unknown)"

    steps_per_hour="unknown"
    eta_seconds="unknown"
    eta_utc="unknown"
    remaining_steps="unknown"

    if [[ "$step" =~ ^[0-9]+$ ]] && [[ "$step" -lt "$TARGET_STEPS" ]]; then
        remaining_steps=$((TARGET_STEPS - step))
    fi

    if [[ "$step" =~ ^[0-9]+$ ]] && [[ "$prev_ts" =~ ^[0-9]+$ ]] && [[ "$prev_step" =~ ^[0-9]+$ ]]; then
        if [[ $prev_ts -gt 0 ]] && [[ $now_epoch -gt $prev_ts ]] && [[ $step -gt $prev_step ]]; then
            delta_steps=$((step - prev_step))
            delta_secs=$((now_epoch - prev_ts))
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

    if [[ -n "$ema_sps" ]] && awk -v s="$ema_sps" 'BEGIN { exit !(s > 0) }'; then
        steps_per_hour="$(awk -v s="$ema_sps" 'BEGIN { printf "%.1f", s * 3600 }')"
        if [[ "$remaining_steps" =~ ^[0-9]+$ ]]; then
            eta_seconds="$(awk -v rem="$remaining_steps" -v s="$ema_sps" 'BEGIN { printf "%.0f", rem / s }')"
            if [[ "$eta_seconds" =~ ^[0-9]+$ ]]; then
                eta_utc="$(date -u -d "@$((now_epoch + eta_seconds))" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
            fi
        fi
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
    } >"$STATUS_FILE"

    prev_ts="$now_epoch"
    prev_step="$step"
    {
        echo "prev_ts=$prev_ts"
        echo "prev_step=$prev_step"
        echo "ema_sps=$ema_sps"
        echo "speed_source=$speed_source"
    } >"$ETA_STATE_FILE"

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
