#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
SYSTEMD_USER_DIR="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
FORGE_ENV_DIR="${FORGE_ENV_DIR:-$HOME/.config/forge}"
FORGE_ENV_FILE="${FORGE_ENV_FILE:-$FORGE_ENV_DIR/v100_training.env}"

mkdir -p "$SYSTEMD_USER_DIR" "$FORGE_ENV_DIR" "$PROJECT_ROOT/artifacts/logs"

if [[ ! -f "$FORGE_ENV_FILE" ]]; then
    cat >"$FORGE_ENV_FILE" <<'ENVEOF'
# V100 runtime contract
TRAIN_CONFIG=configs/models/turkcell_7b_v100_v3_ultrastable.yaml
FALLBACK_CONFIG=configs/models/turkcell_7b_v100_v3b_fallback.yaml
TARGET_STEPS=8601
SAVE_STEPS=250
ENABLE_RESUME=0
RESUME_AFTER_STEP=500
REQUIRE_WANDB=0
CUDA_VISIBLE_DEVICES=0
ENVEOF
    chmod 600 "$FORGE_ENV_FILE"
fi

install -m 0644 \
    "$PROJECT_ROOT/deploy/systemd/forge-v100-training.service" \
    "$SYSTEMD_USER_DIR/forge-v100-training.service"
install -m 0644 \
    "$PROJECT_ROOT/deploy/systemd/forge-v100-watchdog.service" \
    "$SYSTEMD_USER_DIR/forge-v100-watchdog.service"
install -m 0644 \
    "$PROJECT_ROOT/deploy/systemd/forge-v100-watchdog.timer" \
    "$SYSTEMD_USER_DIR/forge-v100-watchdog.timer"

chmod +x \
    "$PROJECT_ROOT/scripts/start_v100_training.sh" \
    "$PROJECT_ROOT/scripts/start_or_resume_full_training.sh" \
    "$PROJECT_ROOT/scripts/monitor_v100_training.sh" \
    "$PROJECT_ROOT/scripts/monitor_training.sh" \
    "$PROJECT_ROOT/scripts/watchdog_training.sh" \
    "$PROJECT_ROOT/scripts/run_v100_completion.sh"

systemctl --user daemon-reload
systemctl --user enable forge-v100-training.service
systemctl --user enable --now forge-v100-watchdog.timer
systemctl --user restart forge-v100-watchdog.service
systemctl --user --no-pager --lines=20 status forge-v100-training.service || true
systemctl --user --no-pager --lines=20 status forge-v100-watchdog.timer || true
systemctl --user --no-pager --lines=20 status forge-v100-watchdog.service || true

echo "V100 watchdog installed. Env file: $FORGE_ENV_FILE"
