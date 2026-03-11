#!/usr/bin/env python3
"""Run QLoRA fine-tuning.

Usage:
    python scripts/run_training.py --config configs/models/turkcell_7b.yaml
    python scripts/run_training.py --config configs/models/turkcell_7b.yaml --dry-run
    python scripts/run_training.py --config configs/models/turkcell_7b.yaml --max-steps 100
"""

import click

from forge.utils.config import load_training_config
from forge.utils.logging import setup_logging
from forge.utils.runtime_guard import enforce_remote_execution


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Model config YAML.")
@click.option("--dry-run", is_flag=True, help="Load model and LoRA without training.")
@click.option("--max-steps", default=-1, help="Override max training steps (-1 = use epochs).")
@click.option(
    "--resume-from",
    default=None,
    type=click.Path(exists=True),
    help="Optional checkpoint directory to resume from.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--lr-override", default=None, type=float, help="Override learning rate (for recovery).")
def main(
    config: str,
    dry_run: bool,
    max_steps: int,
    resume_from: str | None,
    verbose: bool,
    lr_override: float | None,
) -> None:
    """Run QLoRA fine-tuning on a low-resource language model."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    try:
        enforce_remote_execution("training")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    cfg = load_training_config(config)
    click.echo(f"Model: {cfg.model.name}")
    click.echo(f"LoRA rank: {cfg.lora.r}, alpha: {cfg.lora.alpha}")
    click.echo(f"Training epochs: {cfg.training.num_epochs}")

    if max_steps > 0:
        cfg.training.max_steps = max_steps
        click.echo(f"Max steps override: {max_steps}")
    if resume_from:
        click.echo(f"Resume from checkpoint: {resume_from}")
    if lr_override is not None:
        click.echo(f"Learning rate override: {lr_override}")

    from forge.training.trainer import ForgeTrainer

    trainer = ForgeTrainer(cfg, lr_override=lr_override)

    click.echo("\nLoading model and applying LoRA...")
    trainer.setup()
    click.echo(f"Trainable parameters: {trainer._count_trainable_params():,}")

    if dry_run:
        click.echo("\nDry run complete. Model and LoRA loaded successfully.")
        return

    click.echo("\nStarting training...")
    output_path = trainer.train(resume_from_checkpoint=resume_from)
    click.echo(f"\nTraining complete. Adapter saved to {output_path}")


if __name__ == "__main__":
    main()
