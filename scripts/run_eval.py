#!/usr/bin/env python3
"""Run evaluation benchmarks.

Usage:
    python scripts/run_eval.py --model artifacts/merged/turkcell-7b-turkish-v1
    python scripts/run_eval.py --model Turkcell/Turkcell-LLM-7b-v1 --benchmark perplexity
    python scripts/run_eval.py --model artifacts/merged/turkcell-7b-turkish-v1 --benchmark mmlu_tr
"""

import click

from forge.utils.config import EvalConfig
from forge.utils.logging import setup_logging
from forge.utils.runtime_guard import enforce_remote_execution


@click.command()
@click.option("--model", required=True, help="Path to model or HuggingFace repo.")
@click.option("--benchmark", default=None, help="Benchmark: mmlu_tr, perplexity, generation.")
@click.option("--output-dir", default="artifacts/eval", help="Output directory for results.")
@click.option("--device", default="cuda", help="Device for inference.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def main(model: str, benchmark: str | None, output_dir: str, device: str, verbose: bool) -> None:
    """Run evaluation benchmarks against a model."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    try:
        enforce_remote_execution("evaluation")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    cfg = EvalConfig(model_path=model, output_dir=output_dir, device=device)
    if benchmark:
        cfg.benchmarks = [benchmark]

    click.echo(f"Model: {model}")
    click.echo(f"Benchmarks: {', '.join(cfg.benchmarks)}")
    click.echo()

    from forge.evaluation.evaluator import ForgeEvaluator

    evaluator = ForgeEvaluator(cfg)
    results = evaluator.run_all()

    click.echo("\n--- Results ---")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        click.echo(f"  [{status}] {r.name}: {r.score:.4f} ({r.duration_seconds:.1f}s)")

    passed = sum(1 for r in results if r.passed)
    click.echo(f"\nOverall: {passed}/{len(results)} passed")
    click.echo(f"Report: {output_dir}/report.md")


if __name__ == "__main__":
    main()
