"""Tests for training callbacks (no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock

from forge.training.callbacks import EarlyStoppingOnPlateau


def _make_state(global_step: int = 100) -> MagicMock:
    state = MagicMock()
    state.global_step = global_step
    return state


def _make_control() -> MagicMock:
    control = MagicMock()
    control.should_training_stop = False
    return control


def _make_args() -> MagicMock:
    return MagicMock()


def test_early_stopping_improves() -> None:
    """No stop triggered when loss keeps improving."""
    cb = EarlyStoppingOnPlateau(patience=3, min_delta=0.01)
    args = _make_args()
    state = _make_state()
    control = _make_control()

    for loss in [2.0, 1.5, 1.0, 0.8, 0.5]:
        cb.on_evaluate(args, state, control, metrics={"eval_loss": loss})

    assert control.should_training_stop is False


def test_early_stopping_triggers_on_plateau() -> None:
    """Stop triggers after `patience` evaluations without improvement."""
    cb = EarlyStoppingOnPlateau(patience=3, min_delta=0.01)
    args = _make_args()
    state = _make_state()
    control = _make_control()

    # First eval sets the baseline.
    cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.0})
    assert control.should_training_stop is False

    # 3 evals without improvement (within min_delta).
    for _ in range(3):
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.0})

    assert control.should_training_stop is True


def test_early_stopping_resets_on_improvement() -> None:
    """Counter resets when loss improves after a plateau."""
    cb = EarlyStoppingOnPlateau(patience=3, min_delta=0.01)
    args = _make_args()
    state = _make_state()
    control = _make_control()

    cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.0})
    # 2 stale evals.
    cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.0})
    cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.0})

    # Big improvement resets counter.
    cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.5})
    assert control.should_training_stop is False

    # Now need 3 MORE stale evals to trigger.
    cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.5})
    cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.5})
    assert control.should_training_stop is False

    cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.5})
    assert control.should_training_stop is True


def test_early_stopping_skips_when_no_metrics() -> None:
    """No crash when metrics is None or missing eval_loss."""
    cb = EarlyStoppingOnPlateau(patience=2)
    args = _make_args()
    state = _make_state()
    control = _make_control()

    cb.on_evaluate(args, state, control, metrics=None)
    cb.on_evaluate(args, state, control, metrics={"train_loss": 0.5})
    assert control.should_training_stop is False


def test_min_delta_sensitivity() -> None:
    """Small improvements within min_delta are treated as plateaus."""
    cb = EarlyStoppingOnPlateau(patience=2, min_delta=0.1)
    args = _make_args()
    state = _make_state()
    control = _make_control()

    cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.0})
    # 0.95 is only 0.05 better than 1.0, which is < min_delta=0.1.
    cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.95})
    cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.92})

    assert control.should_training_stop is True
