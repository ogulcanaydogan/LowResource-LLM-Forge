#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
FORGE_ENV_FILE="${FORGE_ENV_FILE:-$HOME/.config/forge/v100_training.env}"

if [[ -f "$FORGE_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$FORGE_ENV_FILE"
fi

cd "$PROJECT_ROOT"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/models/turkcell_7b_v100_v3_ultrastable.yaml}"
FALLBACK_CONFIG="${FALLBACK_CONFIG:-configs/models/turkcell_7b_v100_v3b_fallback.yaml}"
TARGET_STEPS="${TARGET_STEPS:-8601}"
SAVE_STEPS="${SAVE_STEPS:-250}"
ENABLE_RESUME="${ENABLE_RESUME:-0}"
RESUME_FROM="${RESUME_FROM:-}"
REQUIRE_WANDB="${REQUIRE_WANDB:-0}"
LR_OVERRIDE="${LR_OVERRIDE:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG_BASENAME="$(basename "$TRAIN_CONFIG")"
CONFIG_SLUG="${CONFIG_BASENAME%.*}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-artifacts/logs}"
ACTIVE_RUN_FILE="${ACTIVE_RUN_FILE:-artifacts/logs/v100_active_run.env}"
STATUS_FILE="${STATUS_FILE:-artifacts/logs/training_monitor_status.txt}"
PID_FILE="${PID_FILE:-artifacts/logs/training_full.pid}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
HF_HOME_DIR="${HF_HOME_DIR:-$PROJECT_ROOT/.hf_cache}"
HF_DATASETS_CACHE_DIR="${HF_DATASETS_CACHE_DIR:-$HF_HOME_DIR/datasets}"
HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-$HF_HOME_DIR/hub}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "${TRAIN_RUN_DIR:-}" ]]; then
    case "$TRAIN_CONFIG" in
        *turkcell_7b_v100_v3b_fallback.yaml)
            TRAIN_RUN_DIR="artifacts/training/turkcell-7b-sft-v100-v3b-fallback"
            ;;
        *turkcell_7b_v100_v3_ultrastable.yaml)
            TRAIN_RUN_DIR="artifacts/training/turkcell-7b-sft-v100-v3-ultrastable"
            ;;
        *turkcell_7b_v100_v2b_fallback.yaml)
            TRAIN_RUN_DIR="artifacts/training/turkcell-7b-sft-v100-v2b-fallback"
            ;;
        *)
            TRAIN_RUN_DIR="artifacts/training/turkcell-7b-sft-v100-v2-stable"
            ;;
    esac
fi

TRAIN_LOG="${TRAIN_LOG:-$TRAIN_LOG_DIR/training_${CONFIG_SLUG}_${RUN_ID}.log}"

abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$PROJECT_ROOT/$path"
    fi
}

TRAIN_RUN_DIR_ABS="$(abs_path "$TRAIN_RUN_DIR")"
TRAIN_LOG_ABS="$(abs_path "$TRAIN_LOG")"
ACTIVE_RUN_FILE_ABS="$(abs_path "$ACTIVE_RUN_FILE")"
PID_FILE_ABS="$(abs_path "$PID_FILE")"

mkdir -p \
    "$(dirname "$TRAIN_RUN_DIR_ABS")" \
    "$(dirname "$TRAIN_LOG_ABS")" \
    "$(dirname "$ACTIVE_RUN_FILE_ABS")" \
    "$(dirname "$PID_FILE_ABS")" \
    "$HF_HOME_DIR" \
    "$HF_DATASETS_CACHE_DIR" \
    "$HF_HUB_CACHE_DIR"

if [[ ! -x "$UV_BIN" ]]; then
    echo "UV executable not found: $UV_BIN" >&2
    exit 1
fi

if [[ "$REQUIRE_WANDB" == "1" ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is required for this run (REQUIRE_WANDB=1)." >&2
    exit 1
fi

find_latest_checkpoint() {
    if [[ ! -d "$TRAIN_RUN_DIR_ABS" ]]; then
        return
    fi

    local current_step=""
    if [[ -f "$STATUS_FILE" ]]; then
        current_step="$(grep -E '^step=' "$STATUS_FILE" | tail -n 1 | cut -d '=' -f 2 || true)"
    fi

    if [[ "$current_step" =~ ^[0-9]+$ ]] && [[ "$current_step" -gt 0 ]]; then
        local filtered_checkpoint
        filtered_checkpoint="$(find "$TRAIN_RUN_DIR_ABS" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
            | sed -E 's#(.*checkpoint-)([0-9]+)$#\2 \1\2#' \
            | awk -v s="$current_step" '$1 <= s' \
            | sort -n \
            | tail -n 1 \
            | cut -d ' ' -f 2- || true)"
        if [[ -n "$filtered_checkpoint" ]]; then
            echo "$filtered_checkpoint"
            return
        fi
    fi

    find "$TRAIN_RUN_DIR_ABS" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

resume_from=""
if [[ -n "$RESUME_FROM" ]]; then
    resume_from="$(abs_path "$RESUME_FROM")"
    if [[ ! -d "$resume_from" ]]; then
        echo "RESUME_FROM checkpoint directory not found: $resume_from" >&2
        exit 1
    fi
elif [[ "$ENABLE_RESUME" == "1" ]]; then
    latest_checkpoint="$(find_latest_checkpoint || true)"
    if [[ -n "$latest_checkpoint" ]]; then
        resume_from="$latest_checkpoint"
    fi
fi

cmd=("$UV_BIN" "run" "python" "scripts/run_training.py" "--config" "$TRAIN_CONFIG")
if [[ -n "$resume_from" ]]; then
    cmd+=("--resume-from" "$resume_from")
fi
if [[ -n "$LR_OVERRIDE" ]]; then
    cmd+=("--lr-override" "$LR_OVERRIDE")
fi

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
    echo "[$ts] forge-training-start run_id=$RUN_ID"
    echo "run_id=$RUN_ID"
    echo "train_config=$TRAIN_CONFIG"
    echo "fallback_config=$FALLBACK_CONFIG"
    echo "target_steps=$TARGET_STEPS"
    echo "save_steps=$SAVE_STEPS"
    echo "train_run_dir=$TRAIN_RUN_DIR"
    echo "train_log=$TRAIN_LOG"
    echo "resume_from=${resume_from:-none}"
    echo "resume_override=${RESUME_FROM:-none}"
    echo "enable_resume=$ENABLE_RESUME"
    echo "require_wandb=$REQUIRE_WANDB"
    echo "command=${cmd[*]}"
} >>"$TRAIN_LOG_ABS"

{
    echo "RUN_ID=$RUN_ID"
    echo "TRAIN_CONFIG=$TRAIN_CONFIG"
    echo "FALLBACK_CONFIG=$FALLBACK_CONFIG"
    echo "TARGET_STEPS=$TARGET_STEPS"
    echo "SAVE_STEPS=$SAVE_STEPS"
    echo "TRAIN_RUN_DIR=$TRAIN_RUN_DIR"
    echo "TRAIN_LOG=$TRAIN_LOG"
    echo "RESUME_FROM=${resume_from:-}"
    echo "ENABLE_RESUME=$ENABLE_RESUME"
    echo "REQUIRE_WANDB=$REQUIRE_WANDB"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "LR_OVERRIDE=${LR_OVERRIDE:-}"
} >"$ACTIVE_RUN_FILE_ABS"

echo "$$" >"$PID_FILE_ABS"

exec > >(tee -a "$TRAIN_LOG_ABS") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] forge-training-exec run_id=$RUN_ID"

exec env \
    FORGE_EXECUTION_CONTEXT=remote \
    HF_HOME="$HF_HOME_DIR" \
    HF_DATASETS_CACHE="$HF_DATASETS_CACHE_DIR" \
    HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE_DIR" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "${cmd[@]}"
