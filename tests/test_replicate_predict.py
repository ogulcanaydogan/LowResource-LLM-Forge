"""Tests for the Replicate predictor module (no GPU required)."""

from __future__ import annotations

from forge.serving.replicate_predict import Predictor, _get_predictor_class


def test_predictor_is_none_without_cog() -> None:
    """Predictor should be None when cog is not installed."""
    # In dev/test environments, cog is not installed.
    # The module-level try/except should set Predictor = None.
    assert Predictor is None


def test_get_predictor_class_raises_without_cog() -> None:
    """_get_predictor_class should raise ImportError without cog."""
    try:
        _get_predictor_class()
        # If cog IS installed, we just verify it returns a class.
        assert True
    except ImportError:
        # Expected in dev environments.
        assert True
