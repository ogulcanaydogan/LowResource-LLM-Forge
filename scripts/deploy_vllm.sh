#!/bin/bash
# Deploy and manage vLLM on DGX Spark via user-level systemd.
set -euo pipefail

SPARK_HOST="${SPARK_HOST:-spark-5fc3}"
SPARK_USER="${SPARK_USER:-weezboo}"
DEPLOY_DIR="/home/${SPARK_USER}/llm-forge"
SERVICE_NAME="forge-vllm"
SYSTEMD_USER_DIR="/home/${SPARK_USER}/.config/systemd/user"
REMOTE_MODEL_DIR="${DEPLOY_DIR}/model"
REMOTE_CONFIG_PATH="${DEPLOY_DIR}/configs/vllm.yaml"
DEFAULT_MODEL_DIR="artifacts/merged/turkcell-7b-turkish-v1"
DEFAULT_CONFIG="configs/serving/vllm_dgx.yaml"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/deploy_vllm.sh deploy [MODEL_DIR] [CONFIG]
  bash scripts/deploy_vllm.sh start
  bash scripts/deploy_vllm.sh stop
  bash scripts/deploy_vllm.sh restart
  bash scripts/deploy_vllm.sh status
  bash scripts/deploy_vllm.sh logs

Examples:
  bash scripts/deploy_vllm.sh deploy artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_dgx.yaml
  bash scripts/deploy_vllm.sh status
EOF
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Missing required command: ${cmd}" >&2
        exit 1
    fi
}

trim_quotes() {
    local value="$1"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    printf '%s' "${value}"
}

read_yaml_value() {
    local key="$1"
    local default="$2"
    local config_path="$3"

    local raw
    raw="$(grep -E "^${key}:" "${config_path}" | head -n1 | cut -d':' -f2- || true)"
    raw="$(echo "${raw}" | xargs || true)"
    raw="$(trim_quotes "${raw}")"
    if [ -z "${raw}" ]; then
        echo "${default}"
        return
    fi
    echo "${raw}"
}

remote_exec() {
    ssh "${SPARK_USER}@${SPARK_HOST}" "$@"
}

systemctl_user() {
    remote_exec "systemctl --user $*"
}

write_service_file() {
    local host="$1"
    local port="$2"
    local tensor_parallel="$3"
    local gpu_memory_utilization="$4"
    local max_model_len="$5"
    local dtype="$6"
    local enable_prefix_caching="$7"

    local prefix_flag=""
    if [ "${enable_prefix_caching}" = "true" ]; then
        prefix_flag="--enable-prefix-caching"
    fi

    local tmp_service
    tmp_service="$(mktemp)"
    cat > "${tmp_service}" <<EOF
[Unit]
Description=LowResource-LLM-Forge vLLM Server
After=network.target

[Service]
Type=simple
WorkingDirectory=${DEPLOY_DIR}
ExecStart=/usr/bin/env python -m vllm.entrypoints.openai.api_server --model ${REMOTE_MODEL_DIR} --host ${host} --port ${port} --tensor-parallel-size ${tensor_parallel} --gpu-memory-utilization ${gpu_memory_utilization} --max-model-len ${max_model_len} --dtype ${dtype} ${prefix_flag}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

    scp "${tmp_service}" "${SPARK_USER}@${SPARK_HOST}:${SYSTEMD_USER_DIR}/${SERVICE_NAME}.service"
    rm -f "${tmp_service}"
}

deploy() {
    local model_dir="${1:-${DEFAULT_MODEL_DIR}}"
    local config="${2:-${DEFAULT_CONFIG}}"

    if [ ! -d "${model_dir}" ]; then
        echo "Model directory not found: ${model_dir}" >&2
        exit 1
    fi
    if [ ! -f "${config}" ]; then
        echo "Config file not found: ${config}" >&2
        exit 1
    fi

    local host
    host="$(read_yaml_value "host" "0.0.0.0" "${config}")"
    local port
    port="$(read_yaml_value "port" "8000" "${config}")"
    local tensor_parallel
    tensor_parallel="$(read_yaml_value "tensor_parallel_size" "1" "${config}")"
    local gpu_memory_utilization
    gpu_memory_utilization="$(read_yaml_value "gpu_memory_utilization" "0.90" "${config}")"
    local max_model_len
    max_model_len="$(read_yaml_value "max_model_len" "4096" "${config}")"
    local dtype
    dtype="$(read_yaml_value "dtype" "float16" "${config}")"
    local enable_prefix_caching
    enable_prefix_caching="$(read_yaml_value "enable_prefix_caching" "true" "${config}")"

    echo "=== Deploying ${SERVICE_NAME} to ${SPARK_USER}@${SPARK_HOST} ==="
    echo "Model dir: ${model_dir}"
    echo "Config: ${config}"

    echo "--- Preparing remote directories ---"
    remote_exec "mkdir -p ${REMOTE_MODEL_DIR} ${DEPLOY_DIR}/configs ${SYSTEMD_USER_DIR}"

    echo "--- Syncing model ---"
    rsync -avz --delete --progress "${model_dir}/" "${SPARK_USER}@${SPARK_HOST}:${REMOTE_MODEL_DIR}/"

    echo "--- Uploading config ---"
    scp "${config}" "${SPARK_USER}@${SPARK_HOST}:${REMOTE_CONFIG_PATH}"

    echo "--- Installing systemd user unit ---"
    write_service_file "${host}" "${port}" "${tensor_parallel}" "${gpu_memory_utilization}" "${max_model_len}" "${dtype}" "${enable_prefix_caching}"

    echo "--- Reloading and starting service ---"
    systemctl_user daemon-reload
    systemctl_user enable --now "${SERVICE_NAME}.service"
    systemctl_user restart "${SERVICE_NAME}.service"
    systemctl_user --no-pager --lines=25 status "${SERVICE_NAME}.service"

    echo ""
    echo "Deployment complete."
    echo "Health check: curl http://${SPARK_HOST}:${port}/health"
    echo "Logs: bash scripts/deploy_vllm.sh logs"
}

start() {
    systemctl_user start "${SERVICE_NAME}.service"
    systemctl_user --no-pager --lines=25 status "${SERVICE_NAME}.service"
}

stop() {
    systemctl_user stop "${SERVICE_NAME}.service"
    systemctl_user --no-pager --lines=25 status "${SERVICE_NAME}.service"
}

restart() {
    systemctl_user restart "${SERVICE_NAME}.service"
    systemctl_user --no-pager --lines=25 status "${SERVICE_NAME}.service"
}

status() {
    systemctl_user --no-pager --lines=25 status "${SERVICE_NAME}.service"
}

logs() {
    local lines="${TAIL_LINES:-200}"
    if [ "${LOG_FOLLOW:-0}" = "1" ]; then
        remote_exec "journalctl --user -u ${SERVICE_NAME}.service -n ${lines} -f"
        return
    fi
    remote_exec "journalctl --user -u ${SERVICE_NAME}.service -n ${lines} --no-pager"
}

main() {
    require_cmd ssh
    require_cmd scp
    require_cmd rsync

    local action="${1:-deploy}"
    shift || true

    case "${action}" in
        deploy) deploy "${1:-${DEFAULT_MODEL_DIR}}" "${2:-${DEFAULT_CONFIG}}" ;;
        start) start ;;
        stop) stop ;;
        restart) restart ;;
        status) status ;;
        logs) logs ;;
        -h|--help|help) usage ;;
        *)
            echo "Unknown action: ${action}" >&2
            usage
            exit 1
            ;;
    esac
}

main "$@"
