"""CLI smoke tests for script --help handling."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_path",
    [
        "scripts/run_training.py",
        "scripts/merge_and_push.py",
        "scripts/run_eval.py",
        "scripts/run_serve.py",
    ],
)
def test_script_help_runs_without_optional_deps(script_path: str) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
    )

    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout

