"""Whisper-based audio transcription for Turkish speech data.

Transcribes audio files to text using OpenAI's Whisper model,
with Turkish language forcing and quality filtering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum confidence threshold for keeping a transcription segment.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Whisper model sizes ordered by VRAM usage (tiny ~1GB, large-v3 ~10GB).
VALID_MODEL_SIZES = ("tiny", "base", "small", "medium", "large", "large-v3")


class WhisperTranscriber:
    """Transcribe audio files to Turkish text using Whisper.

    Produces alpaca-format JSONL where each record is:
        {"instruction": "<source prompt>", "input": "", "output": "<transcript>"}
    """

    def __init__(
        self,
        model_size: str = "medium",
        language: str = "tr",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: str | None = None,
    ) -> None:
        if model_size not in VALID_MODEL_SIZES:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from: {', '.join(VALID_MODEL_SIZES)}"
            )
        self.model_size = model_size
        self.language = language
        self.confidence_threshold = confidence_threshold
        self._device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return self._model

        import whisper

        device = self._device
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "loading_whisper",
            model_size=self.model_size,
            device=device,
        )
        self._model = whisper.load_model(self.model_size, device=device)
        return self._model

    def transcribe_file(self, audio_path: Path) -> dict[str, Any]:
        """Transcribe a single audio file.

        Returns a dict with keys: text, segments, language, avg_logprob.
        """
        model = self._load_model()

        logger.info("transcribing", path=str(audio_path))
        result = model.transcribe(
            str(audio_path),
            language=self.language,
            task="transcribe",
            verbose=False,
        )

        segments = result.get("segments", [])
        avg_logprob = 0.0
        if segments:
            avg_logprob = sum(
                s.get("avg_log_prob", s.get("avg_logprob", 0.0))
                for s in segments
            ) / len(segments)

        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": result.get("language", self.language),
            "avg_logprob": avg_logprob,
        }

    def transcribe_directory(
        self,
        audio_dir: Path,
        output_path: Path,
        extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".ogg"),
    ) -> dict[str, int]:
        """Transcribe all audio files in a directory to JSONL.

        Returns stats dict: total, kept, filtered.
        """
        audio_files = sorted(
            f for f in audio_dir.iterdir()
            if f.suffix.lower() in extensions
        )

        if not audio_files:
            logger.warning("no_audio_files", dir=str(audio_dir))
            return {"total": 0, "kept": 0, "filtered": 0}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {"total": len(audio_files), "kept": 0, "filtered": 0}

        with open(output_path, "w", encoding="utf-8") as f:
            for audio_file in audio_files:
                try:
                    result = self.transcribe_file(audio_file)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "transcription_failed",
                        path=str(audio_file),
                        error=str(exc),
                    )
                    stats["filtered"] += 1
                    continue

                text = result["text"]
                if not text or len(text) < 10:
                    stats["filtered"] += 1
                    continue

                if not self._passes_quality_filter(result):
                    stats["filtered"] += 1
                    continue

                record = {
                    "instruction": f"Transcribe the following audio: {audio_file.name}",
                    "input": "",
                    "output": text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["kept"] += 1

        logger.info("transcription_complete", **stats)
        return stats

    def _passes_quality_filter(self, result: dict[str, Any]) -> bool:
        """Filter low-confidence transcriptions."""
        import math

        avg_logprob = result.get("avg_logprob", 0.0)
        # Convert log probability to approximate confidence.
        confidence = math.exp(avg_logprob) if avg_logprob < 0 else 1.0
        return confidence >= self.confidence_threshold
