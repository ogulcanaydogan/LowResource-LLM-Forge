"""Tests for the Whisper transcriber (no GPU or whisper model required)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from forge.data.whisper_transcriber import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    VALID_MODEL_SIZES,
    WhisperTranscriber,
)


def test_valid_model_sizes() -> None:
    """All expected model sizes are in the valid set."""
    for size in ("tiny", "base", "small", "medium", "large", "large-v3"):
        assert size in VALID_MODEL_SIZES


def test_invalid_model_size_raises() -> None:
    with pytest.raises(ValueError, match="Invalid model_size"):
        WhisperTranscriber(model_size="nonexistent")


def test_default_confidence_threshold() -> None:
    assert DEFAULT_CONFIDENCE_THRESHOLD == 0.6


def test_quality_filter_passes_high_confidence() -> None:
    """High confidence transcription passes the filter."""
    transcriber = WhisperTranscriber(model_size="tiny", confidence_threshold=0.5)
    # avg_logprob of -0.3 => confidence ~ 0.74
    result = {"avg_logprob": -0.3, "text": "Merhaba"}
    assert transcriber._passes_quality_filter(result) is True


def test_quality_filter_rejects_low_confidence() -> None:
    """Low confidence transcription is filtered out."""
    transcriber = WhisperTranscriber(model_size="tiny", confidence_threshold=0.8)
    # avg_logprob of -1.0 => confidence ~ 0.37
    result = {"avg_logprob": -1.0, "text": "..."}
    assert transcriber._passes_quality_filter(result) is False


def test_transcribe_directory_no_audio(tmp_path: Path) -> None:
    """Empty directory returns zero stats."""
    transcriber = WhisperTranscriber(model_size="tiny")
    output = tmp_path / "out.jsonl"
    stats = transcriber.transcribe_directory(tmp_path, output)
    assert stats == {"total": 0, "kept": 0, "filtered": 0}


def test_transcribe_directory_with_mocked_model(tmp_path: Path) -> None:
    """Full directory transcription flow with a mocked whisper model."""
    # Create fake audio files.
    for name in ("clip1.wav", "clip2.wav", "clip3.mp3"):
        (tmp_path / name).write_text("fake audio data")

    transcriber = WhisperTranscriber(
        model_size="tiny",
        language="tr",
        confidence_threshold=0.5,
    )

    # Mock the whisper model.
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "Bu bir test cümlesidir ve yeterince uzundur",
        "segments": [{"avg_log_prob": -0.2}],
        "language": "tr",
    }
    transcriber._model = mock_model

    output = tmp_path / "transcripts.jsonl"
    stats = transcriber.transcribe_directory(tmp_path, output)

    assert stats["total"] == 3
    assert stats["kept"] == 3
    assert stats["filtered"] == 0
    assert output.exists()

    # Verify output is valid JSONL in alpaca format.
    with open(output, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3
    for line in lines:
        rec = json.loads(line)
        assert "instruction" in rec
        assert "output" in rec
        assert len(rec["output"]) > 0


def test_transcribe_directory_filters_short_text(tmp_path: Path) -> None:
    """Transcriptions shorter than 10 chars are filtered."""
    (tmp_path / "short.wav").write_text("fake")

    transcriber = WhisperTranscriber(model_size="tiny")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "Kisa",
        "segments": [],
        "language": "tr",
    }
    transcriber._model = mock_model

    output = tmp_path / "out.jsonl"
    stats = transcriber.transcribe_directory(tmp_path, output)

    assert stats["total"] == 1
    assert stats["kept"] == 0
    assert stats["filtered"] == 1
