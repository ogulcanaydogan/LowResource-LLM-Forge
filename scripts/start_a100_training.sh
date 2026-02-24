#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
cd "$PROJECT_ROOT"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/models/turkcell_7b_a100_v4_recovery.yaml}"
CONFIG_BASENAME="$(basename "$TRAIN_CONFIG")"
CONFIG_SLUG="${CONFIG_BASENAME%.*}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-artifacts/training/${CONFIG_SLUG}}"
TRAIN_LOG="${TRAIN_LOG:-artifacts/logs/training_${CONFIG_SLUG}.log}"
BOOTSTRAP_CHECKPOINT="${BOOTSTRAP_CHECKPOINT:-}"
ENABLE_RESUME="${ENABLE_RESUME:-0}"
HF_HOME_DIR="${HF_HOME_DIR:-$PROJECT_ROOT/.hf_cache}"
HF_DATASETS_CACHE_DIR="${HF_DATASETS_CACHE_DIR:-$HF_HOME_DIR/datasets}"
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-$HF_HOME_DIR/hub}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
REQUIRE_WANDB="${REQUIRE_WANDB:-1}"

mkdir -p \
    "$(dirname "$TRAIN_RUN_DIR")" \
    "$(dirname "$TRAIN_LOG")" \
    "$HF_HOME_DIR" \
    "$HF_DATASETS_CACHE_DIR" \
    "$HF_HUB_CACHE_DIR" \
    artifacts/logs

# Keep a durable per-run log even when systemd unit output targets change.
exec > >(tee -a "$TRAIN_LOG") 2>&1

if [[ ! -x "$UV_BIN" ]]; then
    echo "UV executable not found: $UV_BIN" >&2
    exit 1
fi

if [[ "$REQUIRE_WANDB" == "1" ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is required for this run (REQUIRE_WANDB=1)." >&2
    exit 1
fi

find_latest_checkpoint() {
    if [[ ! -d "$TRAIN_RUN_DIR" ]]; then
        return
    fi
    find "$TRAIN_RUN_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1
}

resume_from=""
if [[ "$ENABLE_RESUME" == "1" ]]; then
    latest_checkpoint="$(find_latest_checkpoint || true)"
    if [[ -n "$latest_checkpoint" ]]; then
        resume_from="$latest_checkpoint"
    elif [[ -n "$BOOTSTRAP_CHECKPOINT" ]] && [[ -d "$BOOTSTRAP_CHECKPOINT" ]]; then
        resume_from="$BOOTSTRAP_CHECKPOINT"
    fi
fi

cmd=("$UV_BIN" "run" "python" "scripts/run_training.py" "--config" "$TRAIN_CONFIG")
if [[ -n "$resume_from" ]]; then
    cmd+=("--resume-from" "$resume_from")
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] forge-training-start"
echo "project_root=$PROJECT_ROOT"
echo "train_config=$TRAIN_CONFIG"
echo "config_slug=$CONFIG_SLUG"
echo "train_run_dir=$TRAIN_RUN_DIR"
echo "train_log=$TRAIN_LOG"
echo "resume_from=${resume_from:-none}"
echo "enable_resume=$ENABLE_RESUME"
echo "require_wandb=$REQUIRE_WANDB"
echo "hf_home=$HF_HOME_DIR"
echo "hf_datasets_cache=$HF_DATASETS_CACHE_DIR"
echo "hf_hub_cache=$HF_HUB_CACHE_DIR"
echo "command=${cmd[*]}"

exec env \
    FORGE_EXECUTION_CONTEXT=remote \
    HF_HOME="$HF_HOME_DIR" \
    HF_DATASETS_CACHE="$HF_DATASETS_CACHE_DIR" \
    HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE_DIR" \
    "${cmd[@]}"
