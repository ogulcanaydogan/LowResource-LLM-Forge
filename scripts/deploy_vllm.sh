#!/bin/bash
# Deploy and manage vLLM on DGX Spark via user-level systemd.
set -euo pipefail

# Backward-compatible aliases:
# - SPARK_HOST / SPARK_USER (legacy)
# - DEPLOY_HOST / DEPLOY_USER (generic)
SPARK_HOST="${SPARK_HOST:-${DEPLOY_HOST:-spark}}"
SPARK_USER="${SPARK_USER:-${DEPLOY_USER:-weezboo}}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
SPARK_SSH_IDENTITY="${SPARK_SSH_IDENTITY:-${DEPLOY_SSH_IDENTITY:-}}"
VLLM_API_KEY="${VLLM_API_KEY:-${FORGE_SERVE_API_KEY:-}}"
ALLOW_INSECURE_SERVE="${ALLOW_INSECURE_SERVE:-0}"
DEPLOY_DIR="/home/${SPARK_USER}/llm-forge"
SERVICE_NAME="forge-vllm"
SYSTEMD_USER_DIR="/home/${SPARK_USER}/.config/systemd/user"
REMOTE_MODELS_DIR="${DEPLOY_DIR}/models"
REMOTE_MODEL_ACTIVE_LINK="${DEPLOY_DIR}/model-active"
REMOTE_CONFIG_PATH="${DEPLOY_DIR}/configs/vllm.yaml"
REMOTE_PYTHON_DEFAULT="/home/${SPARK_USER}/LowResource-LLM-Forge/.venv/bin/python"
DEFAULT_MODEL_DIR="artifacts/merged/turkcell-7b-turkish-v1"
DEFAULT_CONFIG="configs/serving/vllm_spark.yaml"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/deploy_vllm.sh deploy [MODEL_DIR] [CONFIG]
  bash scripts/deploy_vllm.sh reconfigure [CONFIG]
  bash scripts/deploy_vllm.sh set-active <MODEL_NAME_OR_PATH>
  bash scripts/deploy_vllm.sh start
  bash scripts/deploy_vllm.sh stop
  bash scripts/deploy_vllm.sh restart
  bash scripts/deploy_vllm.sh status
  bash scripts/deploy_vllm.sh logs

Examples:
  DEPLOY_HOST=spark VLLM_API_KEY='forge-***' bash scripts/deploy_vllm.sh deploy
  DEPLOY_HOST=100.80.116.20 DEPLOY_SSH_IDENTITY=~/.ssh/id_github_weez bash scripts/deploy_vllm.sh deploy
  DEPLOY_HOST=10.34.9.233 DEPLOY_USER=weezboo SSH_PASSWORD=... bash scripts/deploy_vllm.sh deploy
  bash scripts/deploy_vllm.sh deploy artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_spark.yaml
  bash scripts/deploy_vllm.sh deploy artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_dgx.yaml
  VLLM_API_KEY='forge-***' bash scripts/deploy_vllm.sh reconfigure configs/serving/vllm_spark.yaml
  bash scripts/deploy_vllm.sh set-active model-active
  bash scripts/deploy_vllm.sh set-active /home/weezboo/llm-forge/models/turkcell-7b-recovery-b-20260222-193209
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
    local identity_opts=()
    if [ -n "${SPARK_SSH_IDENTITY}" ]; then
        identity_opts=(-i "${SPARK_SSH_IDENTITY}" -o IdentitiesOnly=yes)
    fi

    if [ -n "${SSH_PASSWORD}" ]; then
        sshpass -p "${SSH_PASSWORD}" ssh \
            "${identity_opts[@]}" \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${SPARK_USER}@${SPARK_HOST}" "$@"
        return
    fi

    ssh "${identity_opts[@]}" "${SPARK_USER}@${SPARK_HOST}" "$@"
}

remote_scp() {
    local src="$1"
    local dst="$2"
    local identity_opts=()
    if [ -n "${SPARK_SSH_IDENTITY}" ]; then
        identity_opts=(-i "${SPARK_SSH_IDENTITY}" -o IdentitiesOnly=yes)
    fi

    if [ -n "${SSH_PASSWORD}" ]; then
        sshpass -p "${SSH_PASSWORD}" scp \
            "${identity_opts[@]}" \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${src}" "${dst}"
        return
    fi

    scp "${identity_opts[@]}" "${src}" "${dst}"
}

remote_rsync() {
    local src="$1"
    local dst="$2"
    local ssh_cmd="ssh"
    if [ -n "${SPARK_SSH_IDENTITY}" ]; then
        ssh_cmd+=" -i ${SPARK_SSH_IDENTITY} -o IdentitiesOnly=yes"
    fi

    if [ -n "${SSH_PASSWORD}" ]; then
        sshpass -p "${SSH_PASSWORD}" rsync \
            -e "${ssh_cmd} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
            -avz --delete --progress \
            "${src}" "${dst}"
        return
    fi

    rsync -e "${ssh_cmd}" -avz --delete --progress "${src}" "${dst}"
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
    local trust_remote_code="$8"
    local enforce_eager="$9"
    local python_bin="${10}"
    local triton_ptxas_path="${11:-}"
    local max_num_seqs="${12:-64}"
    local api_key="${13:-}"

    local prefix_flag=""
    if [ "${enable_prefix_caching}" = "true" ]; then
        prefix_flag="--enable-prefix-caching"
    fi

    local trust_remote_code_flag=""
    if [ "${trust_remote_code}" = "true" ]; then
        trust_remote_code_flag="--trust-remote-code"
    fi

    local enforce_eager_flag=""
    if [ "${enforce_eager}" = "true" ]; then
        enforce_eager_flag="--enforce-eager"
    fi

    local api_key_flag=""
    if [ -n "${api_key}" ]; then
        api_key_flag="--api-key ${api_key}"
    fi

    local triton_env_line=""
    if [ -n "${triton_ptxas_path}" ]; then
        triton_env_line="Environment=TRITON_PTXAS_PATH=${triton_ptxas_path}"
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
${triton_env_line}
ExecStart=${python_bin} -m vllm.entrypoints.openai.api_server --model ${REMOTE_MODEL_ACTIVE_LINK} --host ${host} --port ${port} --tensor-parallel-size ${tensor_parallel} --gpu-memory-utilization ${gpu_memory_utilization} --max-model-len ${max_model_len} --max-num-seqs ${max_num_seqs} --dtype ${dtype} ${prefix_flag} ${trust_remote_code_flag} ${enforce_eager_flag} ${api_key_flag}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

    remote_scp "${tmp_service}" "${SPARK_USER}@${SPARK_HOST}:${SYSTEMD_USER_DIR}/${SERVICE_NAME}.service"
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
    local model_basename
    model_basename="$(basename "${model_dir%/}")"
    if [ -z "${model_basename}" ]; then
        echo "Could not determine model basename from: ${model_dir}" >&2
        exit 1
    fi
    local remote_model_dir
    remote_model_dir="${REMOTE_MODELS_DIR}/${model_basename}"

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
    local trust_remote_code
    trust_remote_code="$(read_yaml_value "trust_remote_code" "false" "${config}")"
    local enforce_eager
    enforce_eager="$(read_yaml_value "enforce_eager" "false" "${config}")"
    local python_bin
    python_bin="$(read_yaml_value "python_bin" "${REMOTE_PYTHON_DEFAULT}" "${config}")"
    local triton_ptxas_path
    triton_ptxas_path="$(read_yaml_value "triton_ptxas_path" "" "${config}")"
    local max_num_seqs
    max_num_seqs="$(read_yaml_value "max_num_seqs" "64" "${config}")"

    if [[ "${host}" != "127.0.0.1" && "${host}" != "localhost" && "${ALLOW_INSECURE_SERVE}" != "1" && -z "${VLLM_API_KEY}" ]]; then
        echo "Refusing insecure deploy: host=${host} requires VLLM_API_KEY." >&2
        echo "Set VLLM_API_KEY (or FORGE_SERVE_API_KEY) or ALLOW_INSECURE_SERVE=1 to bypass." >&2
        exit 1
    fi

    echo "=== Deploying ${SERVICE_NAME} to ${SPARK_USER}@${SPARK_HOST} ==="
    echo "Model dir: ${model_dir}"
    echo "Remote model dir: ${remote_model_dir}"
    echo "Remote active link: ${REMOTE_MODEL_ACTIVE_LINK}"
    echo "Config: ${config}"
    echo "Python: ${python_bin}"
    if [ -n "${VLLM_API_KEY}" ]; then
        echo "Auth: enabled (api key required)"
    else
        echo "Auth: disabled (insecure)"
    fi

    echo "--- Preparing remote directories ---"
    remote_exec "mkdir -p ${remote_model_dir} ${REMOTE_MODELS_DIR} ${DEPLOY_DIR}/configs ${SYSTEMD_USER_DIR}"

    echo "--- Syncing model ---"
    remote_rsync "${model_dir}/" "${SPARK_USER}@${SPARK_HOST}:${remote_model_dir}/"

    echo "--- Updating active model symlink ---"
    remote_exec "ln -sfn \"${remote_model_dir}\" \"${REMOTE_MODEL_ACTIVE_LINK}\" && readlink -f \"${REMOTE_MODEL_ACTIVE_LINK}\""

    echo "--- Uploading config ---"
    remote_scp "${config}" "${SPARK_USER}@${SPARK_HOST}:${REMOTE_CONFIG_PATH}"

    echo "--- Installing systemd user unit ---"
    write_service_file \
        "${host}" \
        "${port}" \
        "${tensor_parallel}" \
        "${gpu_memory_utilization}" \
        "${max_model_len}" \
        "${dtype}" \
        "${enable_prefix_caching}" \
        "${trust_remote_code}" \
        "${enforce_eager}" \
        "${python_bin}" \
        "${triton_ptxas_path}" \
        "${max_num_seqs}" \
        "${VLLM_API_KEY}"

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

reconfigure() {
    local config="${1:-${DEFAULT_CONFIG}}"
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
    local trust_remote_code
    trust_remote_code="$(read_yaml_value "trust_remote_code" "false" "${config}")"
    local enforce_eager
    enforce_eager="$(read_yaml_value "enforce_eager" "false" "${config}")"
    local python_bin
    python_bin="$(read_yaml_value "python_bin" "${REMOTE_PYTHON_DEFAULT}" "${config}")"
    local triton_ptxas_path
    triton_ptxas_path="$(read_yaml_value "triton_ptxas_path" "" "${config}")"
    local max_num_seqs
    max_num_seqs="$(read_yaml_value "max_num_seqs" "64" "${config}")"

    if [[ "${host}" != "127.0.0.1" && "${host}" != "localhost" && "${ALLOW_INSECURE_SERVE}" != "1" && -z "${VLLM_API_KEY}" ]]; then
        echo "Refusing insecure deploy: host=${host} requires VLLM_API_KEY." >&2
        echo "Set VLLM_API_KEY (or FORGE_SERVE_API_KEY) or ALLOW_INSECURE_SERVE=1 to bypass." >&2
        exit 1
    fi

    echo "=== Reconfiguring ${SERVICE_NAME} on ${SPARK_USER}@${SPARK_HOST} ==="
    echo "Config: ${config}"
    echo "Python: ${python_bin}"
    if [ -n "${VLLM_API_KEY}" ]; then
        echo "Auth: enabled (api key required)"
    else
        echo "Auth: disabled (insecure)"
    fi

    remote_exec "mkdir -p ${SYSTEMD_USER_DIR}"
    write_service_file \
        "${host}" \
        "${port}" \
        "${tensor_parallel}" \
        "${gpu_memory_utilization}" \
        "${max_model_len}" \
        "${dtype}" \
        "${enable_prefix_caching}" \
        "${trust_remote_code}" \
        "${enforce_eager}" \
        "${python_bin}" \
        "${triton_ptxas_path}" \
        "${max_num_seqs}" \
        "${VLLM_API_KEY}"

    systemctl_user daemon-reload
    systemctl_user enable --now "${SERVICE_NAME}.service"
    systemctl_user restart "${SERVICE_NAME}.service"
    systemctl_user --no-pager --lines=25 status "${SERVICE_NAME}.service"
}

set_active() {
    local model_ref="${1:-}"
    if [ -z "${model_ref}" ]; then
        echo "Usage: bash scripts/deploy_vllm.sh set-active <MODEL_NAME_OR_PATH>" >&2
        exit 1
    fi

    local target_path
    if [[ "${model_ref}" = /* ]]; then
        target_path="${model_ref}"
    else
        target_path="${REMOTE_MODELS_DIR}/${model_ref}"
    fi

    echo "=== Switching active model on ${SPARK_USER}@${SPARK_HOST} ==="
    echo "Target: ${target_path}"

    remote_exec "if [ ! -d \"${target_path}\" ]; then echo 'Model path not found: ${target_path}' >&2; exit 1; fi"
    remote_exec "if [ ! -f \"${target_path}/config.json\" ]; then echo 'Missing config.json in model path: ${target_path}' >&2; exit 1; fi"
    remote_exec "if [ ! -f \"${target_path}/tokenizer.json\" ] && [ ! -f \"${target_path}/tokenizer.model\" ]; then echo 'Missing tokenizer.json or tokenizer.model in model path: ${target_path}' >&2; exit 1; fi"
    remote_exec "set -- \"${target_path}\"/model*.safetensors; if [ ! -e \"\$1\" ]; then echo 'Missing model*.safetensors in model path: ${target_path}' >&2; exit 1; fi"
    remote_exec "ln -sfn \"${target_path}\" \"${REMOTE_MODEL_ACTIVE_LINK}\" && readlink -f \"${REMOTE_MODEL_ACTIVE_LINK}\""

    echo "--- Restarting service ---"
    systemctl_user restart "${SERVICE_NAME}.service"
    systemctl_user --no-pager --lines=25 status "${SERVICE_NAME}.service"
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
    if [ -n "${SSH_PASSWORD}" ]; then
        require_cmd sshpass
    fi

    local action="${1:-deploy}"
    shift || true

    case "${action}" in
        deploy) deploy "${1:-${DEFAULT_MODEL_DIR}}" "${2:-${DEFAULT_CONFIG}}" ;;
        reconfigure) reconfigure "${1:-${DEFAULT_CONFIG}}" ;;
        set-active) set_active "${1:-}" ;;
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
