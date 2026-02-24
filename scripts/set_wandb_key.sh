#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-$HOME/.config/forge/training.env}"
START_SERVICES="${START_SERVICES:-1}"

key="${1:-}"
if [[ -z "$key" ]]; then
    read -r -s -p "Enter WANDB_API_KEY: " key
    echo
fi

if [[ -z "$key" ]]; then
    echo "WANDB_API_KEY cannot be empty." >&2
    exit 1
fi

mkdir -p "$(dirname "$ENV_FILE")"
tmp_file="$(mktemp)"

if [[ -f "$ENV_FILE" ]]; then
    grep -v '^WANDB_API_KEY=' "$ENV_FILE" >"$tmp_file" || true
fi
printf 'WANDB_API_KEY=%s\n' "$key" >>"$tmp_file"

install -m 600 "$tmp_file" "$ENV_FILE"
rm -f "$tmp_file"

echo "Updated $ENV_FILE"

if [[ "$START_SERVICES" == "1" ]]; then
    systemctl --user daemon-reload
    systemctl --user restart forge-training.service
    systemctl --user restart forge-training-watchdog.service
    systemctl --user restart forge-training-monitor.service
    systemctl --user --no-pager --lines=10 status forge-training.service || true
fi
