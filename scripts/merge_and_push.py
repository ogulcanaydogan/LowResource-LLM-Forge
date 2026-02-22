#!/usr/bin/env python3
"""Merge LoRA adapters and optionally push to HuggingFace Hub.

Usage:
    python scripts/merge_and_push.py \
        --base-model TURKCELL/Turkcell-LLM-7b-v1 \
        --adapter artifacts/training/turkcell-7b-sft-v1/final \
        --output artifacts/merged/turkcell-7b-turkish-v1

    python scripts/merge_and_push.py \
        --base-model TURKCELL/Turkcell-LLM-7b-v1 \
        --adapter artifacts/training/turkcell-7b-sft-v1/final \
        --output artifacts/merged/turkcell-7b-turkish-v1 \
        --push --hub-repo ogulcanaydogan/turkcell-7b-turkish-sft
"""

import click

from forge.utils.logging import setup_logging
from forge.utils.runtime_guard import enforce_remote_execution


@click.command()
@click.option("--base-model", required=True, help="Base model name on HuggingFace.")
@click.option("--adapter", required=True, type=click.Path(exists=True), help="LoRA adapter path.")
@click.option("--output", required=True, help="Output path for merged model.")
@click.option("--push", is_flag=True, help="Push to HuggingFace Hub.")
@click.option("--hub-repo", default=None, help="HF Hub repository (e.g., user/model-name).")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def main(
    base_model: str, adapter: str, output: str, push: bool, hub_repo: str | None, verbose: bool
) -> None:
    """Merge LoRA adapters into base model for deployment."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    try:
        enforce_remote_execution("merge")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Base model: {base_model}")
    click.echo(f"Adapter: {adapter}")
    click.echo(f"Output: {output}")

    from forge.training.merge import LoRAMerger

    merger = LoRAMerger(
        base_model_name=base_model,
        adapter_path=adapter,
        output_path=output,
    )
    merger.merge(push_to_hub=push, hub_repo=hub_repo)

    click.echo(f"\nMerged model saved to {output}")
    if push and hub_repo:
        click.echo(f"Pushed to https://huggingface.co/{hub_repo}")


if __name__ == "__main__":
    main()
