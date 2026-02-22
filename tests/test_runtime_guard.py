"""Tests for runtime safety guard."""

from __future__ import annotations

import pytest

from forge.utils.runtime_guard import (
    enforce_remote_execution,
    is_remote_execution_context,
    should_block_local_execution,
)


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "FORGE_ALLOW_LOCAL",
        "FORGE_EXECUTION_CONTEXT",
        "CI",
        "SSH_CONNECTION",
        "SSH_CLIENT",
        "SSH_TTY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_context_blocks_local_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    assert is_remote_execution_context() is False
    assert should_block_local_execution() is True
    with pytest.raises(RuntimeError, match="FORGE_ALLOW_LOCAL=1"):
        enforce_remote_execution("evaluation")


def test_local_override_allows_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("FORGE_ALLOW_LOCAL", "1")
    assert should_block_local_execution() is False
    enforce_remote_execution("training")


def test_ssh_context_allows_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("SSH_CONNECTION", "127.0.0.1 12345 127.0.0.1 22")
    assert is_remote_execution_context() is True
    assert should_block_local_execution() is False


def test_explicit_remote_context_allows_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("FORGE_EXECUTION_CONTEXT", "remote")
    assert is_remote_execution_context() is True
    enforce_remote_execution("serve")

