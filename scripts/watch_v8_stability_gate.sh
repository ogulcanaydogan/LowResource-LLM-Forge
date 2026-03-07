#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
cd "$PROJECT_ROOT"

ENV_FILE="${ENV_FILE:-$HOME/.config/forge/training.env}"
STATUS_FILE="${STATUS_FILE:-artifacts/logs/training_monitor_status_a100.txt}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GATE_STEPS="${GATE_STEPS:-300}"

FALLBACK_CONFIG="${FALLBACK_CONFIG:-configs/models/turkcell_7b_a100_v8b_ultra_stable_fallback.yaml}"
FALLBACK_RUN_DIR="${FALLBACK_RUN_DIR:-artifacts/training/turkcell-7b-sft-v8b-a100-bf16-ultra-stable-fallback}"
FALLBACK_LOG="${FALLBACK_LOG:-artifacts/logs/training_turkcell_7b_a100_v8b_ultra_stable_fallback.log}"

SCRIPT_LOG="${SCRIPT_LOG:-artifacts/logs/v8_stability_gate.log}"
mkdir -p "$(dirname "$SCRIPT_LOG")"
touch "$SCRIPT_LOG"

if [[ ! -f "$STATUS_FILE" ]]; then
    echo "status_file_missing path=$STATUS_FILE" | tee -a "$SCRIPT_LOG"
    exit 1
fi

current_step="$(awk -F '=' '$1=="step" {print $2}' "$STATUS_FILE" | tail -n 1 || true)"
if [[ ! "$current_step" =~ ^[0-9]+$ ]]; then
    current_step=0
fi
start_step="$current_step"
target_step=$((start_step + GATE_STEPS))

current_log="$(awk -F '=' '$1=="TRAIN_LOG" {print $2}' "$ENV_FILE" | tail -n 1 || true)"
if [[ -z "$current_log" ]]; then
    current_log="artifacts/logs/training_turkcell_7b_a100_v8_stable_reset.log"
fi

marker_line=1
if [[ -f "$current_log" ]]; then
    latest_marker="$(grep -a -n "forge-training-start" "$current_log" | tail -n 1 | cut -d ':' -f 1 || true)"
    if [[ "$latest_marker" =~ ^[0-9]+$ ]] && [[ "$latest_marker" -gt 0 ]]; then
        marker_line=$((latest_marker + 1))
    fi
fi

echo "gate_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) start_step=$start_step target_step=$target_step log=$current_log marker_line=$marker_line" | tee -a "$SCRIPT_LOG"

apply_fallback() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gate_fallback_triggered_utc=$ts reason=$1" | tee -a "$SCRIPT_LOG"
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "env_file_missing path=$ENV_FILE" | tee -a "$SCRIPT_LOG"
        return 1
    fi

    python3 - "$ENV_FILE" "$FALLBACK_CONFIG" "$FALLBACK_RUN_DIR" "$FALLBACK_LOG" <<'PY'
import sys
from pathlib import Path

env_file = Path(sys.argv[1])
fallback_config = sys.argv[2]
fallback_run_dir = sys.argv[3]
fallback_log = sys.argv[4]

raw = env_file.read_text(encoding="utf-8", errors="ignore").splitlines()
pairs = []
seen = set()
for line in raw:
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in seen:
            pairs.append(key)
            seen.add(key)

updates = {
    "TRAIN_CONFIG": fallback_config,
    "TRAIN_RUN_DIR": fallback_run_dir,
    "TRAIN_LOG": fallback_log,
    "ENABLE_RESUME": "0",
    "SAVE_STEPS": "250",
}

for key in updates:
    if key not in seen:
        pairs.append(key)
        seen.add(key)

kv = {}
for line in raw:
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        kv[key.strip()] = value.strip()
for key, value in updates.items():
    kv[key] = value

env_file.write_text("".join(f"{k}={kv.get(k, '')}\n" for k in pairs), encoding="utf-8")
PY

    systemctl --user daemon-reload
    systemctl --user restart forge-training.service forge-training-monitor.service forge-training-watchdog.service
    echo "fallback_service_restart_complete_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SCRIPT_LOG"
}

while true; do
    step="$(awk -F '=' '$1=="step" {print $2}' "$STATUS_FILE" | tail -n 1 || true)"
    state="$(awk -F '=' '$1=="state" {print $2}' "$STATUS_FILE" | tail -n 1 || true)"
    if [[ ! "$step" =~ ^[0-9]+$ ]]; then
        step=0
    fi

    if [[ "$step" -ge "$target_step" ]]; then
        echo "gate_pass_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) step=$step target_step=$target_step" | tee -a "$SCRIPT_LOG"
        exit 0
    fi

    if [[ -f "$current_log" ]]; then
        if tail -n +"$marker_line" "$current_log" | grep -a -q "nan_guard_stopping_training"; then
            apply_fallback "nan_guard_stopping_training_detected"
            exit 0
        fi
    fi

    if [[ "$state" == "stopped" ]]; then
        apply_fallback "training_stopped_before_gate"
        exit 0
    fi

    sleep "$POLL_SECONDS"
done
