#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
cd "$PROJECT_ROOT"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/models/turkcell_7b_a100_v8_stable_reset.yaml}"
RUN_DIR="${RUN_DIR:-artifacts/training/turkcell-7b-sft-v8-a100-bf16-stable-reset}"
TRAIN_LOG="${TRAIN_LOG:-artifacts/logs/training_turkcell_7b_a100_v8_stable_reset.log}"
ADAPTER_DIR="${ADAPTER_DIR:-$RUN_DIR/final}"

BASE_MODEL="${BASE_MODEL:-TURKCELL/Turkcell-LLM-7b-v1}"
MERGED_OUTPUT="${MERGED_OUTPUT:-artifacts/merged/turkcell-7b-a100-v8-stable-reset}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-artifacts/eval/turkcell-7b-a100-v8-stable-reset}"

PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
HUB_REPO="${HUB_REPO:-}"

SERVE_BASE_URL="${SERVE_BASE_URL:-}"
SERVE_API_KEY="${SERVE_API_KEY:-}"
SERVE_CONFIG="${SERVE_CONFIG:-configs/serving/vllm_a100_v8_merged.yaml}"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-240}"
AUTO_START_SERVE="${AUTO_START_SERVE:-1}"
BENCHMARK_NUM_REQUESTS="${BENCHMARK_NUM_REQUESTS:-50}"
BENCHMARK_CONCURRENCY="${BENCHMARK_CONCURRENCY:-5}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-artifacts/benchmarks/turkcell-7b-a100-v8}"
BENCHMARK_OUTPUT="${BENCHMARK_OUTPUT:-$BENCHMARK_OUTPUT_DIR/benchmark_$(date -u +%Y%m%dT%H%M%SZ).json}"
SERVE_LOG="${SERVE_LOG:-artifacts/logs/posttrain_serve_v8.log}"

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

mkdir -p "$BENCHMARK_OUTPUT_DIR" artifacts/logs

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
run_endpoint_checks() {
    local base_url="$1"
    smoke_cmd=("$UV_BIN" run python scripts/smoke_serve.py --base-url "$base_url")
    bench_cmd=(
        "$UV_BIN" run python scripts/benchmark_openai_endpoint.py
        --base-url "$base_url"
        --num-requests "$BENCHMARK_NUM_REQUESTS"
        --concurrency "$BENCHMARK_CONCURRENCY"
        --output "$BENCHMARK_OUTPUT"
    )
    if [[ -n "$SERVE_API_KEY" ]]; then
        smoke_cmd+=(--api-key "$SERVE_API_KEY")
        bench_cmd+=(--api-key "$SERVE_API_KEY")
    fi
    "${smoke_cmd[@]}"
    "${bench_cmd[@]}"
}

if [[ -n "$SERVE_BASE_URL" ]]; then
    echo "  - using external endpoint: $SERVE_BASE_URL"
    run_endpoint_checks "$SERVE_BASE_URL"
elif [[ "$AUTO_START_SERVE" == "1" ]]; then
    if [[ ! -f "$SERVE_CONFIG" ]]; then
        echo "Serving config not found: $SERVE_CONFIG" >&2
        exit 1
    fi

    serve_host="$(grep -E '^host:' "$SERVE_CONFIG" | head -n 1 | cut -d ':' -f2- | tr -d ' "' || true)"
    serve_port="$(grep -E '^port:' "$SERVE_CONFIG" | head -n 1 | cut -d ':' -f2- | tr -d ' ' || true)"
    if [[ -z "$serve_host" ]]; then
        serve_host="127.0.0.1"
    fi
    if [[ "$serve_host" == "0.0.0.0" ]] || [[ "$serve_host" == "::" ]]; then
        serve_host="127.0.0.1"
    fi
    if [[ -z "$serve_port" ]]; then
        serve_port="18020"
    fi
    serve_base_url="http://${serve_host}:${serve_port}"

    serve_cmd=(
        "$UV_BIN" run python scripts/run_serve.py
        --config "$SERVE_CONFIG"
        --no-wait
        --timeout "$SERVE_TIMEOUT"
    )
    if [[ -n "$SERVE_API_KEY" ]]; then
        serve_cmd+=(--api-key "$SERVE_API_KEY")
    fi

    "${serve_cmd[@]}" >"$SERVE_LOG" 2>&1 &
    serve_pid="$!"
    cleanup_serve() {
        if [[ -n "${serve_pid:-}" ]] && kill -0 "$serve_pid" >/dev/null 2>&1; then
            kill -INT "$serve_pid" >/dev/null 2>&1 || true
            sleep 1
            if kill -0 "$serve_pid" >/dev/null 2>&1; then
                kill "$serve_pid" >/dev/null 2>&1 || true
            fi
            wait "$serve_pid" >/dev/null 2>&1 || true
        fi
    }
    trap cleanup_serve EXIT

    ready="0"
    for _ in $(seq 1 "$SERVE_TIMEOUT"); do
        if curl -fsS "${serve_base_url}/health" >/dev/null 2>&1; then
            ready="1"
            break
        fi
        sleep 1
    done
    if [[ "$ready" != "1" ]]; then
        echo "vLLM did not become healthy in ${SERVE_TIMEOUT}s (${serve_base_url})" >&2
        exit 1
    fi

    echo "  - started local vLLM endpoint: $serve_base_url"
    run_endpoint_checks "$serve_base_url"
    cleanup_serve
    trap - EXIT
else
    echo "  - SERVE_BASE_URL not set and AUTO_START_SERVE=0; skipping serve smoke + benchmark"
fi

echo
echo "benchmark_output=$BENCHMARK_OUTPUT"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] post-training-pipeline-complete"
