#!/usr/bin/env python3
"""Publish a merged model to HuggingFace Hub with auto-generated model card.

Combines merge metadata, training config, and evaluation results into a
comprehensive model card, then pushes the full model to HF Hub.

Usage:
    python scripts/publish_to_hub.py \
        --model-dir artifacts/merged/turkcell-7b-turkish-v1 \
        --hub-repo ogulcanaydogan/turkcell-7b-turkish-sft \
        --training-config configs/models/turkcell_7b.yaml

    # With eval results:
    python scripts/publish_to_hub.py \
        --model-dir artifacts/merged/turkcell-7b-turkish-v1 \
        --hub-repo ogulcanaydogan/turkcell-7b-turkish-sft \
        --training-config configs/models/turkcell_7b.yaml \
        --eval-results artifacts/eval/results.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from forge.utils.logging import setup_logging
from forge.utils.runtime_guard import enforce_remote_execution


def _load_merge_info(model_dir: Path) -> dict[str, Any]:
    """Load merge_info.json if it exists."""
    path = model_dir / "merge_info.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_eval_results(eval_path: Path | None) -> dict[str, Any] | None:
    """Load evaluation results.json if provided."""
    if not eval_path or not eval_path.exists():
        return None
    with open(eval_path) as f:
        return json.load(f)


def _load_training_config_raw(config_path: Path | None) -> dict[str, Any] | None:
    """Load raw training config for display."""
    if not config_path or not config_path.exists():
        return None
    from forge.utils.config import load_yaml_config

    return load_yaml_config(config_path)


def _detect_precision(train_cfg: dict[str, Any]) -> str:
    """Return precision label from training config flags."""
    if train_cfg.get("fp16"):
        return "fp16"
    if train_cfg.get("bf16"):
        return "bf16"
    return "fp32"


def generate_model_card(
    hub_repo: str,
    model_dir: Path,
    merge_info: dict[str, Any],
    training_config: dict[str, Any] | None,
    eval_results: dict[str, Any] | None,
    language: str,
) -> str:
    """Generate a HuggingFace model card (README.md) with YAML front matter."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    base_model = merge_info.get("base_model", "unknown")

    # YAML front matter
    lines = [
        "---",
        f"language: {language}",
        "license: apache-2.0",
        "library_name: transformers",
        "tags:",
        "  - text-generation",
        "  - fine-tuned",
        "  - qlora",
        "  - low-resource",
        f"base_model: {base_model}",
        "pipeline_tag: text-generation",
        "---",
        "",
        f"# {hub_repo.split('/')[-1]}",
        "",
        f"Fine-tuned language model for **{language.upper()}** text generation, "
        "built with [LowResource-LLM-Forge]"
        "(https://github.com/ogulcanaydogan/LowResource-LLM-Forge).",
        "",
    ]

    # Model details
    lines.extend([
        "## Model Details",
        "",
        f"- **Base model:** [{base_model}](https://huggingface.co/{base_model})",
        "- **Fine-tuning method:** QLoRA (4-bit quantization + LoRA adapters)",
        f"- **Merge method:** {merge_info.get('merge_method', 'peft_merge_and_unload')}",
        f"- **Published:** {timestamp}",
        "",
    ])

    # Training configuration
    if training_config:
        model_cfg = training_config.get("model", {})
        lora_cfg = training_config.get("lora", {})
        train_cfg = training_config.get("training", {})
        quant_cfg = training_config.get("quantization", {})

        lines.extend([
            "## Training Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Max sequence length | {model_cfg.get('max_seq_length', 'N/A')} |",
            f"| LoRA rank (r) | {lora_cfg.get('r', 'N/A')} |",
            f"| LoRA alpha | {lora_cfg.get('alpha', 'N/A')} |",
            f"| LoRA dropout | {lora_cfg.get('dropout', 'N/A')} |",
            f"| Learning rate | {train_cfg.get('learning_rate', 'N/A')} |",
            f"| Epochs | {train_cfg.get('num_epochs', 'N/A')} |",
            f"| Batch size | {train_cfg.get('per_device_train_batch_size', 'N/A')} |",
            f"| Grad accumulation | {train_cfg.get('gradient_accumulation_steps', 'N/A')} |",
            f"| Scheduler | {train_cfg.get('lr_scheduler_type', 'N/A')} |",
            f"| Precision | {_detect_precision(train_cfg)} |",
            f"| Quantization | {'NF4 4-bit' if quant_cfg.get('load_in_4bit') else 'none'} |",
            "",
        ])

        target_modules = lora_cfg.get("target_modules", [])
        if target_modules:
            lines.extend([
                f"**LoRA target modules:** `{', '.join(target_modules)}`",
                "",
            ])

    # Evaluation results
    if eval_results:
        benchmarks = eval_results.get("benchmarks", [])
        if benchmarks:
            lines.extend([
                "## Evaluation Results",
                "",
                "| Benchmark | Score | Status |",
                "|-----------|-------|--------|",
            ])
            for bench in benchmarks:
                status = "PASS" if bench.get("passed") else "FAIL"
                score = bench.get("score", 0)
                lines.append(f"| {bench['name']} | {score:.4f} | {status} |")
            lines.append("")

            summary = eval_results.get("summary", {})
            total = summary.get("total", 0)
            passed = summary.get("passed", 0)
            lines.extend([
                f"**Overall: {passed}/{total} benchmarks passed**",
                "",
            ])

    # Usage
    lines.extend([
        "## Usage",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'model = AutoModelForCausalLM.from_pretrained("{hub_repo}")',
        f'tokenizer = AutoTokenizer.from_pretrained("{hub_repo}")',
        "",
        "prompt = \"### Instruction:\\nBilgi ver.\\n\\n### Response:\\n\"",
        "inputs = tokenizer(prompt, return_tensors=\"pt\").to(model.device)",
        "outputs = model.generate(**inputs, max_new_tokens=256)",
        "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
        "```",
        "",
    ])

    # vLLM serving
    lines.extend([
        "### vLLM Serving",
        "",
        "```bash",
        f"python -m vllm.entrypoints.openai.api_server --model {hub_repo}",
        "```",
        "",
    ])

    # Framework
    lines.extend([
        "## Framework",
        "",
        "Built with [LowResource-LLM-Forge]"
        "(https://github.com/ogulcanaydogan/LowResource-LLM-Forge) — "
        "a sovereign LLM fine-tuning pipeline for low-resource languages.",
        "",
    ])

    return "\n".join(lines)


@click.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to merged model directory.",
)
@click.option(
    "--hub-repo",
    required=True,
    help="HuggingFace Hub repo (e.g., ogulcanaydogan/turkcell-7b-turkish-sft).",
)
@click.option(
    "--training-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Training config YAML (for model card).",
)
@click.option(
    "--eval-results",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Evaluation results.json (for model card).",
)
@click.option(
    "--language",
    default="tr",
    show_default=True,
    help="ISO language code for model card metadata.",
)
@click.option("--private", is_flag=True, help="Create private repository.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def main(
    model_dir: Path,
    hub_repo: str,
    training_config: Path | None,
    eval_results: Path | None,
    language: str,
    private: bool,
    verbose: bool,
) -> None:
    """Publish a merged model to HuggingFace Hub with auto-generated model card."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    try:
        enforce_remote_execution("publish")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    from huggingface_hub import HfApi

    merge_info = _load_merge_info(model_dir)
    raw_config = _load_training_config_raw(training_config)
    eval_data = _load_eval_results(eval_results)

    click.echo(f"Model directory: {model_dir}")
    click.echo(f"Hub repository:  {hub_repo}")
    click.echo(f"Base model:      {merge_info.get('base_model', 'unknown')}")

    # Generate model card
    card_content = generate_model_card(
        hub_repo=hub_repo,
        model_dir=model_dir,
        merge_info=merge_info,
        training_config=raw_config,
        eval_results=eval_data,
        language=language,
    )

    # Write model card to model directory
    card_path = model_dir / "README.md"
    card_path.write_text(card_content)
    click.echo(f"Generated model card: {card_path}")

    # Upload to Hub
    api = HfApi()

    click.echo(f"\nCreating repository: {hub_repo}")
    api.create_repo(repo_id=hub_repo, private=private, exist_ok=True)

    click.echo("Uploading model files...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=hub_repo,
        commit_message="Upload merged model from LowResource-LLM-Forge",
    )

    url = f"https://huggingface.co/{hub_repo}"
    click.echo(f"\nPublished to: {url}")


if __name__ == "__main__":
    main()
