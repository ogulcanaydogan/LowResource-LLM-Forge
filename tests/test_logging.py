"""Tests for logging setup."""

from __future__ import annotations

from forge.utils.logging import get_logger, setup_logging


def test_setup_logging_console_mode() -> None:
    setup_logging(level="INFO", json_output=False)
    logger = get_logger("tests.logging.console")
    logger.info("console_log_emitted", check="ok")


def test_setup_logging_json_mode() -> None:
    setup_logging(level="INFO", json_output=True)
    logger = get_logger("tests.logging.json")
    logger.info("json_log_emitted", check="ok")

