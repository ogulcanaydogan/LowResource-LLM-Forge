"""Early stopping callback for SFT training."""

from __future__ import annotations

import math
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
    """Stop training when NaN/Inf metrics appear repeatedly."""

    def __init__(
        self,
        consecutive_limit: int = 5,
        watch_keys: tuple[str, ...] = ("loss", "grad_norm", "eval_loss"),
    ) -> None:
        self.consecutive_limit = consecutive_limit
        self.watch_keys = watch_keys
        self._consecutive_hits = 0

    def _handle_metrics(
        self,
        *,
        metrics: dict[str, object] | None,
        state: Any,
        control: Any,
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
        self._handle_metrics(metrics=logs, state=state, control=control, source="log")

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        self._handle_metrics(metrics=metrics, state=state, control=control, source="eval")
