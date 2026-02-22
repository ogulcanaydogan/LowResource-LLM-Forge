"""CLI entrypoint for the LowResource-LLM-Forge pipeline."""

from __future__ import annotations

import click

from forge.utils.logging import setup_logging


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--json-logs", is_flag=True, help="Output logs as JSON.")
def main(verbose: bool, json_logs: bool) -> None:
    """LowResource-LLM-Forge: Fine-tune LLMs for low-resource languages."""
    setup_logging(level="DEBUG" if verbose else "INFO", json_output=json_logs)


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Data config YAML.")
@click.option("--limit", default=0, help="Limit number of samples (0 = all).")
def download(config: str, limit: int) -> None:
    """Download and preprocess training data."""
    from forge.data.collector import DataCollector
    from forge.data.dataset_builder import DatasetBuilder
    from forge.data.preprocessor import DataPreprocessor
    from forge.utils.config import load_data_config

    cfg = load_data_config(config)
    collector = DataCollector(cfg)
    raw_paths = collector.collect_all(limit=limit)

    preprocessor = DataPreprocessor(cfg.preprocessing)
    processed_paths = []
    for path in raw_paths:
        out = path.with_suffix(".clean.jsonl")
        preprocessor.process_file(path, out)
        processed_paths.append(out)

    builder = DatasetBuilder(cfg.output)
    builder.build_sft_dataset(processed_paths)
    click.echo("Data pipeline complete.")


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Model config YAML.")
@click.option("--dry-run", is_flag=True, help="Load model and LoRA without training.")
@click.option("--max-steps", default=-1, help="Override max training steps.")
def train(config: str, dry_run: bool, max_steps: int) -> None:
    """Run QLoRA fine-tuning."""
    from forge.training.trainer import ForgeTrainer
    from forge.utils.config import load_training_config

    cfg = load_training_config(config)
    if max_steps > 0:
        cfg.training.max_steps = max_steps

    trainer = ForgeTrainer(cfg)
    trainer.setup()

    if dry_run:
        click.echo("Dry run complete. Model and LoRA loaded successfully.")
        return

    output_path = trainer.train()
    click.echo(f"Training complete. Adapter saved to {output_path}")


@main.command()
@click.option("--model", required=True, help="Path to model or HF repo.")
@click.option("--benchmark", default=None, help="Run a specific benchmark only.")
@click.option("--output-dir", default="artifacts/eval", help="Output directory.")
def evaluate(model: str, benchmark: str | None, output_dir: str) -> None:
    """Run evaluation benchmarks."""
    from forge.evaluation.evaluator import ForgeEvaluator
    from forge.utils.config import EvalConfig

    cfg = EvalConfig(model_path=model, output_dir=output_dir)
    if benchmark:
        cfg.benchmarks = [benchmark]

    evaluator = ForgeEvaluator(cfg)
    results = evaluator.run_all()

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        click.echo(f"  [{status}] {r.name}: {r.score:.4f}")


@main.command()
@click.option("--base-model", required=True, help="Base model name on HF.")
@click.option("--adapter", required=True, type=click.Path(exists=True), help="LoRA adapter path.")
@click.option("--output", required=True, help="Output path for merged model.")
@click.option("--push", is_flag=True, help="Push to HuggingFace Hub.")
@click.option("--hub-repo", default=None, help="HF Hub repository name.")
def merge(base_model: str, adapter: str, output: str, push: bool, hub_repo: str | None) -> None:
    """Merge LoRA adapters into base model."""
    from forge.training.merge import LoRAMerger

    merger = LoRAMerger(
        base_model_name=base_model,
        adapter_path=adapter,
        output_path=output,
    )
    merger.merge(push_to_hub=push, hub_repo=hub_repo)
    click.echo(f"Merged model saved to {output}")


if __name__ == "__main__":
    main()
