"""CLI entrypoint. Heavy imports are lazy to keep --help fast."""

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
        cfg.training.max_steps = max_steps  # useful for quick sanity checks

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


@main.command()
@click.option(
    "--audio-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing audio files.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output JSONL path (default: <audio-dir>/transcriptions.jsonl).",
)
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=True),
    help="Whisper config YAML.",
)
@click.option("--model-size", default="medium", help="Whisper model size.")
@click.option("--language", default="tr", help="Target language code.")
def transcribe(
    audio_dir: str,
    output: str | None,
    config: str | None,
    model_size: str,
    language: str,
) -> None:
    """Transcribe audio files using Whisper."""
    from pathlib import Path

    from forge.data.whisper_transcriber import WhisperTranscriber

    if config:
        import yaml

        with open(config) as f:
            cfg = yaml.safe_load(f)
        model_size = cfg.get("model_size", model_size)
        language = cfg.get("language", language)

    audio_path = Path(audio_dir)
    output_path = Path(output) if output else audio_path / "transcriptions.jsonl"

    transcriber = WhisperTranscriber(
        model_size=model_size,
        language=language,
    )
    stats = transcriber.transcribe_directory(audio_path, output_path)
    click.echo(f"Transcription complete: {stats}")


@main.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True),
    help="Merged model directory.",
)
@click.option("--hub-repo", required=True, help="HF Hub repo (user/name).")
@click.option(
    "--training-config",
    default=None,
    type=click.Path(exists=True),
    help="Training YAML (for model card).",
)
@click.option(
    "--eval-results",
    default=None,
    type=click.Path(exists=True),
    help="Eval results JSON (for model card).",
)
@click.option("--language", default="tr", help="Primary language code.")
def publish(
    model_dir: str,
    hub_repo: str,
    training_config: str | None,
    eval_results: str | None,
    language: str,
) -> None:
    """Publish a merged model to HuggingFace Hub."""
    import subprocess
    import sys

    from forge.utils.runtime_guard import enforce_remote_execution

    # uploads GBs of weights, don't run from laptop
    enforce_remote_execution("publish")

    cmd = [
        sys.executable,
        "scripts/publish_to_hub.py",
        "--model-dir",
        model_dir,
        "--hub-repo",
        hub_repo,
        "--language",
        language,
    ]
    if training_config:
        cmd.extend(["--training-config", training_config])
    if eval_results:
        cmd.extend(["--eval-results", eval_results])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        click.echo(f"Publish failed:\n{result.stderr}", err=True)
        raise SystemExit(1)
    click.echo(result.stdout)
    click.echo(f"Model published to https://huggingface.co/{hub_repo}")


@main.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Serving config YAML.",
)
def serve(config: str) -> None:
    """Start a vLLM inference server."""
    from forge.serving.vllm_server import VLLMServer
    from forge.utils.config import load_serving_config
    from forge.utils.runtime_guard import enforce_remote_execution

    # vLLM needs GPU, block accidental local runs
    enforce_remote_execution("serve")

    cfg = load_serving_config(config)
    server = VLLMServer(cfg)
    click.echo(f"Starting vLLM server at {server.base_url} ...")
    server.start()


@main.command()
@click.option("--base-url", required=True, help="vLLM base URL.")
@click.option("--api-key", default=None, help="API key for the endpoint.")
@click.option(
    "--num-requests", default=50, help="Number of requests for benchmark."
)
@click.option(
    "--concurrency", default=5, help="Concurrent request count."
)
def benchmark(
    base_url: str,
    api_key: str | None,
    num_requests: int,
    concurrency: int,
) -> None:
    """Benchmark an OpenAI-compatible endpoint."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/benchmark_openai_endpoint.py",
        "--base-url",
        base_url,
        "--num-requests",
        str(num_requests),
        "--concurrency",
        str(concurrency),
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        click.echo(f"Benchmark failed:\n{result.stderr}", err=True)
        raise SystemExit(1)
    click.echo(result.stdout)


if __name__ == "__main__":
    main()
