#!/usr/bin/env python3
"""Generate a deterministic training manifest for a completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from forge.utils.config import load_training_config

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _extract_log_times(log_file: Path) -> tuple[str, str]:
    if not log_file.exists():
        return "unknown", "unknown"

    start_ts = "unknown"
    end_ts = "unknown"
    with log_file.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "training_started" in line and start_ts == "unknown":
                match = TIMESTAMP_RE.match(line.strip())
                if match:
                    start_ts = match.group(1)
            if "training_complete" in line or "Training complete. Adapter saved to" in line:
                match = TIMESTAMP_RE.match(line.strip())
                if match:
                    end_ts = match.group(1)
    return start_ts, end_ts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training manifest JSON.")
    parser.add_argument("--config", required=True, help="Training config path.")
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument("--log-file", required=True, help="Training log file path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output manifest path (defaults to <run-dir>/manifest.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()
    log_file = Path(args.log_file).resolve()
    output_path = Path(args.output).resolve() if args.output else run_dir / "manifest.json"

    cfg = load_training_config(config_path)
    train_path = Path(cfg.train_data_path).resolve()
    eval_path = Path(cfg.eval_data_path).resolve()

    final_dir = run_dir / "final"
    checkpoints = sorted(p.name for p in run_dir.glob("checkpoint-*") if p.is_dir())
    start_ts, end_ts = _extract_log_times(log_file)

    manifest = {
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "run_dir": str(run_dir),
        "log_file": str(log_file),
        "run_start_utc": start_ts,
        "run_end_utc": end_ts,
        "model_name": cfg.model.name,
        "run_name": cfg.wandb.run_name,
        "train_data_path": str(train_path),
        "eval_data_path": str(eval_path),
        "train_records": _line_count(train_path),
        "eval_records": _line_count(eval_path),
        "final_dir_exists": final_dir.exists(),
        "checkpoint_dirs": checkpoints,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    output_path.write_text(payload, encoding="utf-8")
    print(f"manifest_written={output_path}")


if __name__ == "__main__":
    main()
