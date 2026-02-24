#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
SYSTEMD_USER_DIR="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
FORGE_ENV_DIR="${FORGE_ENV_DIR:-$HOME/.config/forge}"
FORGE_ENV_FILE="${FORGE_ENV_FILE:-$FORGE_ENV_DIR/training.env}"

mkdir -p "$SYSTEMD_USER_DIR" "$PROJECT_ROOT/artifacts/logs" "$FORGE_ENV_DIR"

if [[ ! -f "$FORGE_ENV_FILE" ]]; then
    cat >"$FORGE_ENV_FILE" <<'EOF'
# Required for training with WandB.
# Set your real key before starting forge-training.service.
WANDB_API_KEY=

# Optional overrides:
# TRAIN_CONFIG=configs/models/turkcell_7b_a100_v4_recovery.yaml
# TRAIN_RUN_DIR=artifacts/training/turkcell-7b-sft-v4-a100-bf16-recovery
# ENABLE_RESUME=0
# REQUIRE_WANDB=1
EOF
    chmod 600 "$FORGE_ENV_FILE"
fi

install -m 0644 \
    "$PROJECT_ROOT/deploy/systemd/forge-training.service" \
    "$SYSTEMD_USER_DIR/forge-training.service"
install -m 0644 \
    "$PROJECT_ROOT/deploy/systemd/forge-training-watchdog.service" \
    "$SYSTEMD_USER_DIR/forge-training-watchdog.service"
install -m 0644 \
    "$PROJECT_ROOT/deploy/systemd/forge-training-monitor.service" \
    "$SYSTEMD_USER_DIR/forge-training-monitor.service"

chmod +x \
    "$PROJECT_ROOT/scripts/start_a100_training.sh" \
    "$PROJECT_ROOT/scripts/monitor_a100_training.sh" \
    "$PROJECT_ROOT/scripts/training_watchdog.py"

systemctl --user daemon-reload
systemctl --user enable forge-training.service
systemctl --user enable forge-training-watchdog.service
systemctl --user enable forge-training-monitor.service

if grep -qE '^WANDB_API_KEY=.+$' "$FORGE_ENV_FILE"; then
    systemctl --user restart forge-training.service
    systemctl --user restart forge-training-watchdog.service
    systemctl --user restart forge-training-monitor.service
else
    systemctl --user stop forge-training-monitor.service || true
    systemctl --user stop forge-training-watchdog.service || true
    systemctl --user stop forge-training.service || true
fi

systemctl --user --no-pager --lines=20 status forge-training.service || true
systemctl --user --no-pager --lines=20 status forge-training-watchdog.service || true
systemctl --user --no-pager --lines=20 status forge-training-monitor.service || true
echo
echo "Edit $FORGE_ENV_FILE and set WANDB_API_KEY before starting training."
echo "Or run: scripts/set_wandb_key.sh"
