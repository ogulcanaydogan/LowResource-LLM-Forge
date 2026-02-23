#!/bin/bash
# Print a compact runtime dashboard for Spark + VM330 + GitHub runner.
set -euo pipefail

SPARK_TARGET="${SPARK_TARGET:-spark}"
VM330_TARGET="${VM330_TARGET:-weezboo@10.34.9.233}"
SPARK_SSH_IDENTITY="${SPARK_SSH_IDENTITY:-}"
VM330_SSH_IDENTITY="${VM330_SSH_IDENTITY:-}"
RUNNER_NAME="${RUNNER_NAME:-vm330-gpu-runner}"
REPO_SLUG="${REPO_SLUG:-ogulcanaydogan/LowResource-LLM-Forge}"

ssh_run() {
    local target="$1"
    local identity="$2"
    local remote_cmd="$3"
    local opts=(-o BatchMode=yes -o ConnectTimeout=8)
    if [ -n "${identity}" ]; then
        opts+=(-i "${identity}" -o IdentitiesOnly=yes)
    fi
    ssh "${opts[@]}" "${target}" "${remote_cmd}"
}

print_host_block() {
    local label="$1"
    local target="$2"
    local identity="${3:-}"

    echo "[$label]"
    if ! ssh_run "${target}" "${identity}" "echo ok" >/dev/null 2>&1; then
        echo "  reachable: no"
        echo
        return
    fi

    local payload
    payload="$(ssh_run "${target}" "${identity}" '
svc="$(systemctl --user is-active forge-vllm.service 2>/dev/null || echo unknown)"
smoke_timer="$(systemctl --user is-active forge-smoke-check.timer 2>/dev/null || echo unknown)"
health="$(curl --max-time 8 -s -o /dev/null -w "%{http_code}" http://127.0.0.1:18000/health || true)"
model="$(systemctl --user cat forge-vllm.service 2>/dev/null | sed -n "s/^[[:space:]]*ExecStart=.*--model \\([^[:space:]]*\\).*/\\1/p" | tail -n1 || echo unknown)"
smoke_last="$(tail -n 1 ~/llm-forge/monitor/checks.log 2>/dev/null || echo none)"
alert_last="$(tail -n 1 ~/llm-forge/monitor/alerts.log 2>/dev/null || echo none)"
printf "service=%s\nsmoke_timer=%s\nhealth=%s\nmodel=%s\nsmoke_last=%s\nalert_last=%s\n" \
  "$svc" "$smoke_timer" "$health" "$model" "$smoke_last" "$alert_last"
')"

    while IFS= read -r line; do
        echo "  ${line}"
    done <<<"${payload}"
    echo
}

echo "Forge Runtime Dashboard"
echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

print_host_block "spark" "${SPARK_TARGET}" "${SPARK_SSH_IDENTITY}"
print_host_block "vm330" "${VM330_TARGET}" "${VM330_SSH_IDENTITY}"

echo "[github-runner]"
if command -v gh >/dev/null 2>&1; then
    gh_info="$(gh api "repos/${REPO_SLUG}/actions/runners" --jq ".runners[] | select(.name==\"${RUNNER_NAME}\") | \"status=\\(.status) busy=\\(.busy) labels=\\([.labels[].name] | join(\",\"))\"" 2>/dev/null || true)"
    if [ -n "${gh_info}" ]; then
        echo "  ${gh_info}"
    else
        echo "  status=unknown busy=unknown labels=unknown"
    fi
    latest_eval="$(gh run list --workflow eval-gate.yml --limit 1 2>/dev/null | head -n 1 || true)"
    if [ -n "${latest_eval}" ]; then
        echo "  latest_eval=${latest_eval}"
    fi
else
    echo "  gh_cli=missing"
fi
