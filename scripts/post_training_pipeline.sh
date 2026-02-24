#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
cd "$PROJECT_ROOT"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/models/turkcell_7b_a100_v4_recovery.yaml}"
RUN_DIR="${RUN_DIR:-artifacts/training/turkcell-7b-sft-v4-a100-bf16-recovery}"
TRAIN_LOG="${TRAIN_LOG:-artifacts/logs/training_a100_bf16_v4_recovery.log}"
ADAPTER_DIR="${ADAPTER_DIR:-$RUN_DIR/final}"

BASE_MODEL="${BASE_MODEL:-TURKCELL/Turkcell-LLM-7b-v1}"
MERGED_OUTPUT="${MERGED_OUTPUT:-artifacts/merged/turkcell-7b-a100-v4-recovery}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-artifacts/eval/turkcell-7b-a100-v4-recovery}"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HUB_REPO="${HUB_REPO:-}"

SERVE_BASE_URL="${SERVE_BASE_URL:-}"
SERVE_API_KEY="${SERVE_API_KEY:-}"
BENCHMARK_NUM_REQUESTS="${BENCHMARK_NUM_REQUESTS:-50}"
BENCHMARK_CONCURRENCY="${BENCHMARK_CONCURRENCY:-5}"

if [[ ! -x "$UV_BIN" ]]; then
    echo "UV executable not found: $UV_BIN" >&2
    exit 1
fi

if [[ ! -d "$ADAPTER_DIR" ]]; then
    echo "Adapter directory not found: $ADAPTER_DIR" >&2
    exit 1
fi

if [[ "$PUSH_TO_HUB" == "1" ]] && [[ -z "$HUB_REPO" ]]; then
    echo "HUB_REPO is required when PUSH_TO_HUB=1." >&2
    exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] post-training-pipeline-start"
echo "train_config=$TRAIN_CONFIG"
echo "run_dir=$RUN_DIR"
echo "adapter_dir=$ADAPTER_DIR"

echo
echo "[1/4] Generate training manifest"
"$UV_BIN" run python scripts/generate_training_manifest.py \
    --config "$TRAIN_CONFIG" \
    --run-dir "$RUN_DIR" \
    --log-file "$TRAIN_LOG"

echo
echo "[2/4] Run offline evaluations (mmlu_tr, perplexity, generation)"
for bench in mmlu_tr perplexity generation; do
    out_dir="$EVAL_OUTPUT_ROOT/$bench"
    mkdir -p "$out_dir"
    echo "  - benchmark=$bench output=$out_dir"
    "$UV_BIN" run python scripts/run_eval.py \
        --model "$ADAPTER_DIR" \
        --benchmark "$bench" \
        --output-dir "$out_dir"
done

echo
echo "[3/4] Merge adapters into base model"
merge_cmd=(
    "$UV_BIN" run python scripts/merge_and_push.py
    --base-model "$BASE_MODEL"
    --adapter "$ADAPTER_DIR"
    --output "$MERGED_OUTPUT"
)
if [[ "$PUSH_TO_HUB" == "1" ]]; then
    merge_cmd+=(--push --hub-repo "$HUB_REPO")
fi
"${merge_cmd[@]}"

echo
echo "[4/4] Optional serving smoke/benchmark"
if [[ -n "$SERVE_BASE_URL" ]]; then
    smoke_cmd=("$UV_BIN" run python scripts/smoke_serve.py --base-url "$SERVE_BASE_URL")
    bench_cmd=(
        "$UV_BIN" run python scripts/benchmark_openai_endpoint.py
        --base-url "$SERVE_BASE_URL"
        --num-requests "$BENCHMARK_NUM_REQUESTS"
        --concurrency "$BENCHMARK_CONCURRENCY"
    )
    if [[ -n "$SERVE_API_KEY" ]]; then
        smoke_cmd+=(--api-key "$SERVE_API_KEY")
        bench_cmd+=(--api-key "$SERVE_API_KEY")
    fi
    "${smoke_cmd[@]}"
    "${bench_cmd[@]}"
else
    echo "  - SERVE_BASE_URL not set; skipping serve smoke + benchmark"
fi

echo
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] post-training-pipeline-complete"
