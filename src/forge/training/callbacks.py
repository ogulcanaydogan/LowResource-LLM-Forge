"""Training callbacks: early stopping on eval loss plateau."""

from __future__ import annotations

from typing import Any

from forge.utils.logging import get_logger

logger = get_logger(__name__)


class EarlyStoppingOnPlateau:
    """Stop training when eval loss plateaus for `patience` eval steps.

    Compatible with the ``transformers.TrainerCallback`` protocol.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
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
                control.should_training_stop = True
