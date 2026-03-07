#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
FORGE_ENV_FILE="${FORGE_ENV_FILE:-$HOME/.config/forge/v100_training.env}"
MONITOR_STATUS_FILE="${MONITOR_STATUS_FILE:-$PROJECT_ROOT/artifacts/logs/training_monitor_status.txt}"
WATCHDOG_STATUS_FILE="${WATCHDOG_STATUS_FILE:-$PROJECT_ROOT/artifacts/logs/training_watchdog_status.txt}"
SUMMARY_FILE="${SUMMARY_FILE:-$PROJECT_ROOT/artifacts/logs/v100_completion_summary.md}"
LOCK_DIR="${LOCK_DIR:-$PROJECT_ROOT/artifacts/logs/v100_completion.lock.d}"
DONE_FILE="${DONE_FILE:-$PROJECT_ROOT/artifacts/logs/v100_completion.done}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

cd "$PROJECT_ROOT"

if [[ -f "$DONE_FILE" ]]; then
    exit 0
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    exit 0
fi
trap 'rmdir "$LOCK_DIR" >/dev/null 2>&1 || true' EXIT

if [[ ! -f "$MONITOR_STATUS_FILE" ]]; then
    exit 0
fi

target_steps="$(grep -E '^target_steps=' "$MONITOR_STATUS_FILE" | tail -n 1 | cut -d '=' -f 2 || true)"
step="$(grep -E '^step=' "$MONITOR_STATUS_FILE" | tail -n 1 | cut -d '=' -f 2 || true)"
state="$(grep -E '^state=' "$MONITOR_STATUS_FILE" | tail -n 1 | cut -d '=' -f 2 || true)"

if [[ ! "$target_steps" =~ ^[0-9]+$ ]]; then
    target_steps=8601
fi
if [[ ! "$step" =~ ^[0-9]+$ ]]; then
    step=0
fi

if [[ "$state" != "completed" ]] && [[ "$step" -lt "$target_steps" ]]; then
    exit 0
fi

if [[ -f "$FORGE_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$FORGE_ENV_FILE"
fi

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/models/turkcell_7b_v100_v3_ultrastable.yaml}"
RUN_DIR="${RUN_DIR:-artifacts/training/turkcell-7b-sft-v100-v3-ultrastable}"
TRAIN_LOG="${TRAIN_LOG:-artifacts/logs/training_turkcell_7b_v100_v3_ultrastable.log}"
ADAPTER_DIR="${ADAPTER_DIR:-$RUN_DIR/final}"

BASE_MODEL="${BASE_MODEL:-TURKCELL/Turkcell-LLM-7b-v1}"
MERGED_OUTPUT="${MERGED_OUTPUT:-artifacts/merged/turkcell-7b-v100-v3-ultrastable}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-artifacts/eval/turkcell-7b-v100-v3-ultrastable}"
SERVE_CONFIG="${SERVE_CONFIG:-configs/serving/vllm_v100_v3_merged.yaml}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-artifacts/benchmarks/turkcell-7b-v100-v3-ultrastable}"

export UV_BIN
export TRAIN_CONFIG
export RUN_DIR
export TRAIN_LOG
export ADAPTER_DIR
export BASE_MODEL
export MERGED_OUTPUT
export EVAL_OUTPUT_ROOT
export SERVE_CONFIG
export BENCHMARK_OUTPUT_DIR
export PUSH_TO_HUB=0
export AUTO_START_SERVE=1
export FORGE_EXECUTION_CONTEXT=remote

pipeline_status="success"
pipeline_error=""
if ! bash "$PROJECT_ROOT/scripts/post_training_pipeline.sh"; then
    pipeline_status="failed"
    pipeline_error="post_training_pipeline_failed"
fi

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
watchdog_snapshot=""
if [[ -f "$WATCHDOG_STATUS_FILE" ]]; then
    watchdog_snapshot="$(cat "$WATCHDOG_STATUS_FILE")"
fi

mkdir -p "$(dirname "$SUMMARY_FILE")"
cat >"$SUMMARY_FILE" <<EOF
# V100 Completion Summary

- completed_at_utc: ${timestamp}
- final_step: ${step}
- target_steps: ${target_steps}
- pipeline_status: ${pipeline_status}
- train_config: ${TRAIN_CONFIG}
- run_dir: ${RUN_DIR}
- adapter_dir: ${ADAPTER_DIR}
- merged_output: ${MERGED_OUTPUT}
- eval_outputs:
  - ${EVAL_OUTPUT_ROOT}/mmlu_tr
  - ${EVAL_OUTPUT_ROOT}/perplexity
  - ${EVAL_OUTPUT_ROOT}/generation
- serve_config: ${SERVE_CONFIG}
- benchmark_output_dir: ${BENCHMARK_OUTPUT_DIR}

## Watchdog Snapshot
\`\`\`
${watchdog_snapshot}
\`\`\`
EOF

if [[ "$pipeline_status" == "success" ]]; then
    touch "$DONE_FILE"
else
    {
        echo
        echo "## Error"
        echo "${pipeline_error}"
    } >>"$SUMMARY_FILE"
    exit 1
fi
