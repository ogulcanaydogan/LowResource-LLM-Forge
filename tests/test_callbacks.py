"""Tests for training callbacks (no GPU required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from forge.training.callbacks import EarlyStoppingOnPlateau, NaNGuardCallback


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


# --- NaNGuardCallback tests ---


def _make_state_with_history(
    global_step: int = 100,
    lr: float = 1e-5,
) -> MagicMock:
    state = MagicMock()
    state.global_step = global_step
    state.log_history = [{"learning_rate": lr, "loss": 2.5, "step": global_step}]
    return state


def _make_args_with_output(output_dir: str) -> MagicMock:
    args = MagicMock()
    args.output_dir = output_dir
    return args


def test_nan_guard_no_stop_below_limit() -> None:
    """No stop when NaN hits are below the consecutive limit."""
    cb = NaNGuardCallback(consecutive_limit=3)
    args = _make_args()
    state = _make_state()
    control = _make_control()

    cb.on_log(args, state, control, logs={"loss": float("nan")})
    cb.on_log(args, state, control, logs={"loss": float("nan")})

    assert control.should_training_stop is False
    assert cb._consecutive_hits == 2


def test_nan_guard_stops_at_limit() -> None:
    """Training stops when consecutive NaN limit is reached."""
    cb = NaNGuardCallback(consecutive_limit=2)
    args = _make_args()
    state = _make_state()
    control = _make_control()

    cb.on_log(args, state, control, logs={"loss": float("nan")})
    cb.on_log(args, state, control, logs={"loss": float("nan")})

    assert control.should_training_stop is True


def test_nan_guard_resets_on_good_metrics() -> None:
    """Counter resets when a non-NaN metric is logged."""
    cb = NaNGuardCallback(consecutive_limit=3)
    args = _make_args()
    state = _make_state()
    control = _make_control()

    cb.on_log(args, state, control, logs={"loss": float("nan")})
    cb.on_log(args, state, control, logs={"loss": float("nan")})
    assert cb._consecutive_hits == 2

    cb.on_log(args, state, control, logs={"loss": 2.5})
    assert cb._consecutive_hits == 0
    assert control.should_training_stop is False


def test_nan_guard_writes_recovery_file(tmp_path: Path) -> None:
    """Recovery request file is written when limit is reached."""
    recovery_path = str(tmp_path / "recovery.env")
    cb = NaNGuardCallback(consecutive_limit=2, recovery_request_path=recovery_path)

    # Create a fake checkpoint dir
    ckpt_dir = tmp_path / "output" / "checkpoint-500"
    ckpt_dir.mkdir(parents=True)
    args = _make_args_with_output(str(tmp_path / "output"))
    state = _make_state_with_history(global_step=500, lr=1e-5)
    control = _make_control()

    cb.on_log(args, state, control, logs={"loss": float("nan")})
    assert not Path(recovery_path).exists()

    cb.on_log(args, state, control, logs={"loss": float("nan")})
    assert control.should_training_stop is True
    assert Path(recovery_path).exists()

    content = Path(recovery_path).read_text()
    assert "RECOVERY_REQUESTED=1" in content
    assert "CURRENT_LR=1e-05" in content
    assert "checkpoint-500" in content
    assert "NAN_FIELD=loss" in content


def test_nan_guard_no_recovery_file_without_path() -> None:
    """No file written when recovery_request_path is None."""
    cb = NaNGuardCallback(consecutive_limit=1)
    args = _make_args()
    state = _make_state()
    control = _make_control()

    cb.on_log(args, state, control, logs={"loss": float("nan")})
    assert control.should_training_stop is True


def test_nan_guard_detects_inf() -> None:
    """Inf values are also detected."""
    cb = NaNGuardCallback(consecutive_limit=1)
    args = _make_args()
    state = _make_state()
    control = _make_control()

    cb.on_log(args, state, control, logs={"grad_norm": float("inf")})
    assert control.should_training_stop is True
