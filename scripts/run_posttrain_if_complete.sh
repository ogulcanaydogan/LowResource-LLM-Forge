#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
cd "$PROJECT_ROOT"

STATUS_FILE="${STATUS_FILE:-artifacts/logs/training_monitor_status_a100.txt}"
LOCK_FILE="${LOCK_FILE:-artifacts/logs/posttrain_v8.lock}"
DONE_FILE="${DONE_FILE:-artifacts/logs/posttrain_v8.done}"
SUMMARY_FILE="${SUMMARY_FILE:-artifacts/logs/posttrain_v8_summary.md}"
POSTTRAIN_LOG="${POSTTRAIN_LOG:-artifacts/logs/posttrain_v8.log}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/models/turkcell_7b_a100_v8_stable_reset.yaml}"
RUN_DIR="${RUN_DIR:-${TRAIN_RUN_DIR:-artifacts/training/turkcell-7b-sft-v8-a100-bf16-stable-reset}}"
TRAIN_LOG="${TRAIN_LOG:-artifacts/logs/training_turkcell_7b_a100_v8_stable_reset.log}"
ADAPTER_DIR="${ADAPTER_DIR:-$RUN_DIR/final}"
MERGED_OUTPUT="${MERGED_OUTPUT:-artifacts/merged/turkcell-7b-a100-v8-stable-reset}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-artifacts/eval/turkcell-7b-a100-v8-stable-reset}"
SERVE_CONFIG="${SERVE_CONFIG:-configs/serving/vllm_a100_v8_merged.yaml}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-artifacts/benchmarks/turkcell-7b-a100-v8}"
BENCHMARK_OUTPUT="${BENCHMARK_OUTPUT:-$BENCHMARK_OUTPUT_DIR/benchmark_$(date -u +%Y%m%dT%H%M%SZ).json}"

mkdir -p "$(dirname "$LOCK_FILE")" "$(dirname "$DONE_FILE")" "$(dirname "$SUMMARY_FILE")" "$BENCHMARK_OUTPUT_DIR"

resolve_serve_endpoint() {
    local host
    local port
    host="$(grep -E '^host:' "$SERVE_CONFIG" | head -n 1 | cut -d ':' -f2- | tr -d ' "' || true)"
    port="$(grep -E '^port:' "$SERVE_CONFIG" | head -n 1 | cut -d ':' -f2- | tr -d ' ' || true)"
    if [[ -z "$host" ]]; then
        host="127.0.0.1"
    fi
    if [[ "$host" == "0.0.0.0" ]] || [[ "$host" == "::" ]]; then
        host="127.0.0.1"
    fi
    if [[ -z "$port" ]]; then
        port="18020"
    fi
    echo "http://${host}:${port}"
}

collect_eval_status_lines() {
    local bench
    local results_json
    local pass_total
    local status
    for bench in mmlu_tr perplexity generation; do
        results_json="${EVAL_OUTPUT_ROOT}/${bench}/results.json"
        if [[ -f "$results_json" ]]; then
            pass_total="$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])).get("summary",{}); print(f"{int(s.get(\"passed\",0))}/{int(s.get(\"total\",0))}")' "$results_json")"
            status="$(python3 -c 'import json,sys; b=(json.load(open(sys.argv[1])).get("benchmarks") or [{}])[0]; print("PASS" if b.get("passed") else "FAIL")' "$results_json")"
            echo "- ${bench}: ${status} (${pass_total})"
        else
            echo "- ${bench}: MISSING (${results_json})"
        fi
    done
}

if [[ -f "$DONE_FILE" ]]; then
    echo "posttrain_already_done file=$DONE_FILE"
    exit 0
fi

if [[ ! -f "$STATUS_FILE" ]]; then
    echo "posttrain_waiting status_file_missing=$STATUS_FILE"
    exit 0
fi

state="$(awk -F '=' '$1=="state" {print $2}' "$STATUS_FILE" | tail -n 1 || true)"
step="$(awk -F '=' '$1=="step" {print $2}' "$STATUS_FILE" | tail -n 1 || true)"
target_steps="$(awk -F '=' '$1=="target_steps" {print $2}' "$STATUS_FILE" | tail -n 1 || true)"

if [[ ! "$step" =~ ^[0-9]+$ ]]; then
    step=0
fi
if [[ ! "$target_steps" =~ ^[0-9]+$ ]] || [[ "$target_steps" -le 0 ]]; then
    target_steps=8601
fi

if [[ "$state" != "completed" ]] && [[ "$step" -lt "$target_steps" ]]; then
    echo "posttrain_waiting state=${state:-unknown} step=$step target_steps=$target_steps"
    exit 0
fi

if [[ -f "$LOCK_FILE" ]]; then
    locked_pid="$(awk -F '=' '$1=="pid" {print $2}' "$LOCK_FILE" | tail -n 1 || true)"
    if [[ "$locked_pid" =~ ^[0-9]+$ ]] && kill -0 "$locked_pid" >/dev/null 2>&1; then
        echo "posttrain_lock_active pid=$locked_pid file=$LOCK_FILE"
        exit 0
    fi
    rm -f "$LOCK_FILE"
fi

{
    echo "pid=$$"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$LOCK_FILE"

cleanup_lock() {
    rm -f "$LOCK_FILE"
}
trap cleanup_lock EXIT

if [[ -f "$DONE_FILE" ]]; then
    echo "posttrain_already_done file=$DONE_FILE"
    exit 0
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] posttrain_trigger_start" | tee -a "$POSTTRAIN_LOG"
echo "status_state=$state step=$step target_steps=$target_steps" | tee -a "$POSTTRAIN_LOG"
serve_endpoint="$(resolve_serve_endpoint)"

set +e
(
    export TRAIN_CONFIG
    export RUN_DIR
    export TRAIN_LOG
    export ADAPTER_DIR
    export MERGED_OUTPUT
    export EVAL_OUTPUT_ROOT
    export PUSH_TO_HUB=0
    export SERVE_CONFIG
    export AUTO_START_SERVE=1
    export BENCHMARK_OUTPUT_DIR
    export BENCHMARK_OUTPUT
    bash scripts/post_training_pipeline.sh
) >>"$POSTTRAIN_LOG" 2>&1
pipeline_rc=$?
set -e

if [[ "$pipeline_rc" -ne 0 ]]; then
    {
        echo "# Post-Training v8 Summary"
        echo
        echo "- status: FAILED"
        echo "- finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "- pipeline_exit_code: $pipeline_rc"
        echo "- train_config: $TRAIN_CONFIG"
        echo "- run_dir: $RUN_DIR"
        echo "- serve_endpoint: $serve_endpoint"
        echo "- eval_results:"
        collect_eval_status_lines
        echo "- log: $POSTTRAIN_LOG"
    } >"$SUMMARY_FILE"
    echo "posttrain_failed rc=$pipeline_rc log=$POSTTRAIN_LOG summary=$SUMMARY_FILE"
    exit "$pipeline_rc"
fi

{
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "train_config=$TRAIN_CONFIG"
    echo "run_dir=$RUN_DIR"
    echo "merged_output=$MERGED_OUTPUT"
    echo "eval_output_root=$EVAL_OUTPUT_ROOT"
    echo "benchmark_output=$BENCHMARK_OUTPUT"
    echo "serve_endpoint=$serve_endpoint"
    echo "posttrain_log=$POSTTRAIN_LOG"
} >"$DONE_FILE"

{
    echo "# Post-Training v8 Summary"
    echo
    echo "- status: SUCCESS"
    echo "- completed_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- train_config: $TRAIN_CONFIG"
    echo "- run_dir: $RUN_DIR"
    echo "- eval_results:"
    collect_eval_status_lines
    echo "- merged_model: $MERGED_OUTPUT"
    echo "- serve_endpoint: $serve_endpoint"
    echo "- eval_output_root: $EVAL_OUTPUT_ROOT"
    echo "- benchmark_output: $BENCHMARK_OUTPUT"
    echo "- posttrain_log: $POSTTRAIN_LOG"
} >"$SUMMARY_FILE"

echo "posttrain_complete done_file=$DONE_FILE summary=$SUMMARY_FILE"
