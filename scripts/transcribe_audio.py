#!/usr/bin/env python3
"""Transcribe audio files to Turkish text using Whisper.

Usage:
    python scripts/transcribe_audio.py \
        --config configs/data/whisper.yaml \
        --audio-dir /path/to/audio/

    python scripts/transcribe_audio.py \
        --audio-dir /path/to/audio/ \
        --model-size medium \
        --output data/raw/tr/whisper_transcripts.jsonl
"""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from forge.utils.logging import setup_logging
from forge.utils.runtime_guard import enforce_remote_execution


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Whisper config YAML (optional, overrides other flags).",
)
@click.option(
    "--audio-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing audio files.",
)
@click.option("--model-size", default="medium", show_default=True, help="Whisper model size.")
@click.option("--language", default="tr", show_default=True, help="Target language code.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/raw/tr/whisper_transcripts.jsonl"),
    show_default=True,
    help="Output JSONL path.",
)
@click.option("--confidence", default=0.6, show_default=True, type=float, help="Min confidence.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def main(
    config: Path | None,
    audio_dir: Path,
    model_size: str,
    language: str,
    output: Path,
    confidence: float,
    verbose: bool,
) -> None:
    """Transcribe audio files to alpaca-format JSONL using Whisper."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    try:
        enforce_remote_execution("whisper-transcribe")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    # Load config if provided.
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f) or {}
        model_size = cfg.get("model_size", model_size)
        language = cfg.get("language", language)
        confidence = cfg.get("confidence_threshold", confidence)
        output = Path(cfg.get("output_path", str(output)))

    from forge.data.whisper_transcriber import WhisperTranscriber

    transcriber = WhisperTranscriber(
        model_size=model_size,
        language=language,
        confidence_threshold=confidence,
    )

    click.echo(f"Model: whisper-{model_size}")
    click.echo(f"Language: {language}")
    click.echo(f"Audio dir: {audio_dir}")
    click.echo(f"Output: {output}")

    stats = transcriber.transcribe_directory(audio_dir, output)

    click.echo(f"\nTotal files:  {stats['total']}")
    click.echo(f"Kept:         {stats['kept']}")
    click.echo(f"Filtered:     {stats['filtered']}")

    if stats["kept"] > 0:
        click.echo(f"\nTranscripts saved to {output}")
    else:
        click.echo("\nNo transcriptions passed quality filter.")


if __name__ == "__main__":
    main()
