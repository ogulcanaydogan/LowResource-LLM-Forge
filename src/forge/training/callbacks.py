"""Training callbacks: early stopping, NaN guard with self-healing recovery."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - fallback for non-training environments
    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base class when transformers is unavailable."""

        pass

logger = get_logger(__name__)


class EarlyStoppingOnPlateau(TrainerCallback):
    """Stop training when eval loss plateaus for `patience` eval steps.

    Compatible with the ``transformers.TrainerCallback`` protocol.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001) -> None:
        self.patience = patience  # eval steps to wait
        self.min_delta = min_delta  # min improvement to reset counter
        self._best_loss: float | None = None
        self._wait = 0

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if self._best_loss is None or eval_loss < self._best_loss - self.min_delta:
            self._best_loss = eval_loss
            self._wait = 0
            logger.info(
                "eval_loss_improved",
                eval_loss=eval_loss,
                step=state.global_step,
            )
        else:
            self._wait += 1
            logger.info(
                "eval_loss_plateau",
                eval_loss=eval_loss,
                best_loss=self._best_loss,
                wait=self._wait,
                patience=self.patience,
            )
            if self._wait >= self.patience:
                logger.info("early_stopping", step=state.global_step)
                # HF Trainer checks this flag after eval
                control.should_training_stop = True


def _is_non_finite(value: object) -> bool:
    """Return True when a metric value is NaN/Inf."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"nan", "inf", "+inf", "-inf"}
    return False


class NaNGuardCallback(TrainerCallback):
    """Stop training when NaN/Inf metrics appear repeatedly.

    When the consecutive limit is reached, writes a recovery request file
    so the watchdog can restart with a halved learning rate.
    """

    def __init__(
        self,
        consecutive_limit: int = 5,
        watch_keys: tuple[str, ...] = ("loss", "grad_norm", "eval_loss"),
        recovery_request_path: str | None = None,
    ) -> None:
        self.consecutive_limit = consecutive_limit
        self.watch_keys = watch_keys
        self.recovery_request_path = recovery_request_path
        self._consecutive_hits = 0

    def _write_recovery_request(
        self, *, state: Any, args: Any, bad_field: str
    ) -> None:
        """Write a recovery request file for the watchdog to pick up."""
        if not self.recovery_request_path:
            return

        current_lr = 0.0
        if hasattr(state, "log_history") and state.log_history:
            for entry in reversed(state.log_history):
                if "learning_rate" in entry:
                    current_lr = float(entry["learning_rate"])
                    break

        last_checkpoint = ""
        if hasattr(args, "output_dir") and hasattr(state, "global_step"):
            output_dir = Path(args.output_dir)
            checkpoints = sorted(
                output_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
            )
            if checkpoints:
                last_checkpoint = str(checkpoints[-1])

        path = Path(self.recovery_request_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"RECOVERY_REQUESTED=1\n"
            f"CURRENT_LR={current_lr}\n"
            f"LAST_CHECKPOINT={last_checkpoint}\n"
            f"NAN_FIELD={bad_field}\n"
            f"STEP={state.global_step}\n"
            f"TIMESTAMP={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        )
        logger.info(
            "nan_recovery_request_written",
            path=str(path),
            current_lr=current_lr,
            last_checkpoint=last_checkpoint,
        )

    def _handle_metrics(
        self,
        *,
        metrics: dict[str, object] | None,
        state: Any,
        control: Any,
        args: Any,
        source: str,
    ) -> None:
        if not metrics:
            return

        bad_values: dict[str, object] = {
            key: value
            for key, value in metrics.items()
            if key in self.watch_keys and _is_non_finite(value)
        }

        if not bad_values:
            self._consecutive_hits = 0
            return

        self._consecutive_hits += 1
        logger.warning(
            "nan_guard_detected",
            source=source,
            step=state.global_step,
            hits=self._consecutive_hits,
            limit=self.consecutive_limit,
            bad_metrics=bad_values,
        )

        if self._consecutive_hits >= self.consecutive_limit:
            bad_field = next(iter(bad_values))
            self._write_recovery_request(
                state=state, args=args, bad_field=bad_field
            )
            logger.error(
                "nan_guard_stopping_training",
                source=source,
                step=state.global_step,
                limit=self.consecutive_limit,
            )
            control.should_training_stop = True

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        self._handle_metrics(metrics=logs, state=state, control=control, args=args, source="log")

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        self._handle_metrics(metrics=metrics, state=state, control=control, args=args, source="eval")
