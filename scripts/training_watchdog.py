#!/usr/bin/env python3
"""Watchdog for long-running training on remote GPU hosts.

Restarts a user-level systemd training service when:
1) Too many consecutive metric lines contain NaN.
2) Training step does not advance for a configured stall timeout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class WatchdogState:
    """Persisted state between watchdog loops."""

    last_metric_hash: str = ""
    nan_consecutive: int = 0
    last_step: int = 0
    last_step_change_ts: float = 0.0


def _config_slug() -> str:
    train_config = os.getenv("TRAIN_CONFIG", "configs/models/turkcell_7b_a100_v4_recovery.yaml")
    return Path(train_config).stem


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _load_status_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _status_completed(status: dict[str, str], target_steps: int) -> bool:
    state = status.get("state", "")
    try:
        step = int(status.get("step", "0"))
    except ValueError:
        step = 0

    if state == "completed":
        return True
    return step >= target_steps


def parse_args() -> argparse.Namespace:
    slug = _config_slug()
    default_log_file = os.getenv("TRAIN_LOG", f"artifacts/logs/training_{slug}.log")
    default_state_file = os.getenv(
        "TRAIN_WATCHDOG_STATE_FILE",
        f"artifacts/logs/training_watchdog_state_{slug}.json",
    )
    default_status_file = os.getenv(
        "TRAIN_WATCHDOG_STATUS_FILE",
        f"artifacts/logs/training_watchdog_status_{slug}.txt",
    )
    default_monitor_status_file = os.getenv(
        "TRAIN_MONITOR_STATUS_FILE",
        "artifacts/logs/training_monitor_status_a100.txt",
    )
    default_target_steps = _int_env("TARGET_STEPS", 8601)
    default_poll_seconds = _int_env("WATCHDOG_POLL_SECONDS", 60)
    default_stall_seconds = _int_env("WATCHDOG_STALL_SECONDS", 5400)
    default_nan_limit = _int_env("WATCHDOG_NAN_CONSECUTIVE_LIMIT", 5)

    parser = argparse.ArgumentParser(description="Monitor training and auto-restart on failures.")
    parser.add_argument("--service", default="forge-training.service")
    parser.add_argument("--log-file", default=default_log_file)
    parser.add_argument("--state-file", default=default_state_file)
    parser.add_argument("--status-file", default=default_status_file)
    parser.add_argument("--monitor-status-file", default=default_monitor_status_file)
    parser.add_argument("--target-steps", type=int, default=default_target_steps)
    parser.add_argument("--poll-seconds", type=int, default=default_poll_seconds)
    parser.add_argument("--nan-consecutive-limit", type=int, default=default_nan_limit)
    parser.add_argument("--stall-seconds", type=int, default=default_stall_seconds)
    parser.add_argument("--max-read-bytes", type=int, default=2_000_000)
    return parser.parse_args()


def _run_systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        text=True,
        capture_output=True,
    )


def is_service_active(service: str) -> bool:
    proc = _run_systemctl("is-active", "--quiet", service)
    return proc.returncode == 0


def restart_service(service: str) -> bool:
    proc = _run_systemctl("restart", service)
    return proc.returncode == 0


def start_service(service: str) -> bool:
    proc = _run_systemctl("start", service)
    return proc.returncode == 0


def read_tail_text(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="ignore")


def parse_training_tail(text: str, target_steps: int) -> tuple[int, str]:
    if not text:
        return 0, ""

    marker_idx = text.rfind("forge-training-start")
    if marker_idx != -1:
        text = text[marker_idx:]

    step_pattern = re.compile(rf"(\d+)/{target_steps}\b")
    steps = [int(match.group(1)) for match in step_pattern.finditer(text)]
    last_step = steps[-1] if steps else 0

    metric_lines: list[str] = []
    for line in text.splitlines():
        if ("'loss':" in line and "'grad_norm':" in line) or "'eval_loss':" in line:
            metric_lines.append(line)
    last_metric = metric_lines[-1] if metric_lines else ""
    return last_step, last_metric


def metric_hash(metric_line: str) -> str:
    if not metric_line:
        return ""
    return hashlib.sha256(metric_line.encode("utf-8", errors="ignore")).hexdigest()


def load_state(path: Path) -> WatchdogState:
    if not path.exists():
        return WatchdogState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return WatchdogState(
            last_metric_hash=str(payload.get("last_metric_hash", "")),
            nan_consecutive=int(payload.get("nan_consecutive", 0)),
            last_step=int(payload.get("last_step", 0)),
            last_step_change_ts=float(payload.get("last_step_change_ts", 0.0)),
        )
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return WatchdogState()


def save_state(path: Path, state: WatchdogState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def write_status(
    path: Path,
    *,
    service: str,
    active: bool,
    step: int,
    target_steps: int,
    metric_line: str,
    state: WatchdogState,
    action: str,
) -> None:
    pct = int((step * 100) / target_steps) if target_steps > 0 else 0
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status_lines = [
        f"timestamp_utc={timestamp}",
        f"service={service}",
        f"active={'yes' if active else 'no'}",
        f"step={step}",
        f"target_steps={target_steps}",
        f"percent={pct}",
        f"nan_consecutive={state.nan_consecutive}",
        f"last_step_change_ts={int(state.last_step_change_ts)}",
        f"last_metric_contains_nan={'yes' if 'nan' in metric_line.lower() else 'no'}",
        f"action={action}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(status_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file)
    state_file = Path(args.state_file)
    status_file = Path(args.status_file)
    monitor_status_file = Path(args.monitor_status_file)
    state = load_state(state_file)

    while True:
        now = time.time()
        action = "none"
        active = is_service_active(args.service)
        monitor_status = _load_status_file(monitor_status_file)
        training_completed = _status_completed(monitor_status, args.target_steps)

        if not active:
            if training_completed:
                action = "completed_no_restart"
            else:
                started = start_service(args.service)
                action = "start_service" if started else "start_failed"
                active = is_service_active(args.service)

        tail_text = read_tail_text(log_file, args.max_read_bytes)
        step, last_metric = parse_training_tail(tail_text, args.target_steps)
        current_metric_hash = metric_hash(last_metric)

        if step < state.last_step:
            # Service likely restarted and step counter reset.
            state.last_step = step
            state.last_step_change_ts = now
            state.nan_consecutive = 0
            state.last_metric_hash = current_metric_hash
        elif step > state.last_step:
            state.last_step = step
            state.last_step_change_ts = now
        elif state.last_step_change_ts == 0.0 and step > 0:
            state.last_step_change_ts = now

        if current_metric_hash and current_metric_hash != state.last_metric_hash:
            state.last_metric_hash = current_metric_hash
            if "nan" in last_metric.lower():
                state.nan_consecutive += 1
            else:
                state.nan_consecutive = 0

        stalled = (
            active
            and state.last_step_change_ts > 0
            and (now - state.last_step_change_ts) >= args.stall_seconds
        )
        nan_limit_hit = state.nan_consecutive >= args.nan_consecutive_limit

        if (nan_limit_hit or stalled) and not training_completed:
            restarted = restart_service(args.service)
            if nan_limit_hit:
                action = "restart_nan_limit_hit" if restarted else "restart_nan_failed"
            else:
                action = "restart_stall_timeout" if restarted else "restart_stall_failed"
            state.nan_consecutive = 0
            state.last_metric_hash = ""
            state.last_step_change_ts = now
            active = is_service_active(args.service)
        elif training_completed:
            action = "completed_no_restart"

        save_state(state_file, state)
        write_status(
            status_file,
            service=args.service,
            active=active,
            step=step,
            target_steps=args.target_steps,
            metric_line=last_metric,
            state=state,
            action=action,
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
