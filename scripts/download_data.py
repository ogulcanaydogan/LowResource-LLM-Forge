#!/usr/bin/env python3
"""Download and preprocess all data for a language.

Usage:
    python scripts/download_data.py --config configs/data/turkish.yaml
    python scripts/download_data.py --config configs/data/turkish.yaml --limit 1000
"""

import click

from forge.data.collector import DataCollector
from forge.data.dataset_builder import DatasetBuilder
from forge.data.preprocessor import DataPreprocessor
from forge.utils.config import load_data_config
from forge.utils.logging import setup_logging


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Data config YAML.")
@click.option("--limit", default=0, help="Limit samples per source (0 = all).")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def main(config: str, limit: int, verbose: bool) -> None:
    """Download, preprocess, and build training datasets."""
    setup_logging(level="DEBUG" if verbose else "INFO")

    cfg = load_data_config(config)
    click.echo(f"Language: {cfg.language_name} ({cfg.language})")
    click.echo(f"Sources: {len(cfg.sources)}")

    # Step 1: Collect raw data
    click.echo("\n--- Step 1: Collecting data ---")
    collector = DataCollector(cfg)
    raw_paths = collector.collect_all(limit=limit)
    click.echo(f"Downloaded {len(raw_paths)} sources")

    # Step 2: Preprocess
    click.echo("\n--- Step 2: Preprocessing ---")
    preprocessor = DataPreprocessor(cfg.preprocessing)
    processed_paths = []
    for path in raw_paths:
        out = path.with_suffix(".clean.jsonl")
        stats = preprocessor.process_file(path, out)
        click.echo(f"  {path.name}: {stats['kept']}/{stats['total']} kept")
        processed_paths.append(out)

    # Step 3: Build final datasets
    click.echo("\n--- Step 3: Building datasets ---")
    builder = DatasetBuilder(cfg.output)
    stats = builder.build_sft_dataset(processed_paths)
    click.echo(f"  Train: {stats['train']} samples")
    click.echo(f"  Eval:  {stats['eval']} samples")
    click.echo(f"  Total: {stats['total']} samples")

    click.echo("\nData pipeline complete.")


if __name__ == "__main__":
    main()
