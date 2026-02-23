"""Block model-loading commands in local shells.

Prevents accidentally spinning up a 7B model on a MacBook.
"""

from __future__ import annotations

import os

_REMOTE_CONTEXT_VALUES = {"remote", "dgx", "v100", "ci"}  # see deploy scripts
_SSH_MARKERS = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in _TRUE_VALUES)


def is_remote_execution_context() -> bool:
    """Return True when command is executed in a trusted remote context."""
    context = os.getenv("FORGE_EXECUTION_CONTEXT", "").strip().lower()
    if context in _REMOTE_CONTEXT_VALUES:
        return True

    if _is_truthy(os.getenv("CI")):
        return True

    return any(bool(os.getenv(marker)) for marker in _SSH_MARKERS)


def should_block_local_execution() -> bool:
    """Return True when local execution should be blocked."""
    allow_local = os.getenv("FORGE_ALLOW_LOCAL", "").strip() == "1"
    return not allow_local and not is_remote_execution_context()


def enforce_remote_execution(operation: str) -> None:
    """Raise RuntimeError when a model-loading operation runs locally."""
    if not should_block_local_execution():
        return

    raise RuntimeError(
        f"Blocked local {operation} run. "
        "This command is remote-first and only allowed via SSH/CI by default. "
        "Use FORGE_ALLOW_LOCAL=1 to override intentionally."
    )

