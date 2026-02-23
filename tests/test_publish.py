"""Tests for the publish_to_hub model card generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Make the scripts directory importable so we can test generate_model_card.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from publish_to_hub import generate_model_card  # noqa: E402


def test_model_card_basic(tmp_path: Path) -> None:
    """Model card contains required sections with minimal inputs."""
    card = generate_model_card(
        hub_repo="user/my-model",
        model_dir=tmp_path,
        merge_info={"base_model": "org/base-7b", "merge_method": "peft_merge_and_unload"},
        training_config=None,
        eval_results=None,
        language="tr",
    )

    assert "language: tr" in card
    assert "# my-model" in card
    assert "org/base-7b" in card
    assert "QLoRA" in card
    assert "```python" in card
    assert "vLLM" in card


def test_model_card_with_training_config(tmp_path: Path) -> None:
    """Model card includes training config table when provided."""
    training_config = {
        "model": {"max_seq_length": 2048},
        "lora": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "training": {
            "learning_rate": 0.0002,
            "num_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "lr_scheduler_type": "cosine",
            "fp16": True,
            "bf16": False,
        },
        "quantization": {"load_in_4bit": True},
    }

    card = generate_model_card(
        hub_repo="user/my-model",
        model_dir=tmp_path,
        merge_info={"base_model": "org/base-7b"},
        training_config=training_config,
        eval_results=None,
        language="tr",
    )

    assert "Training Configuration" in card
    assert "32" in card  # LoRA rank
    assert "64" in card  # LoRA alpha
    assert "cosine" in card
    assert "q_proj" in card
    assert "NF4 4-bit" in card


def test_model_card_with_eval_results(tmp_path: Path) -> None:
    """Model card includes evaluation results table."""
    eval_results = {
        "benchmarks": [
            {"name": "perplexity", "score": 12.5, "passed": True, "duration_seconds": 30.0},
            {"name": "generation", "score": 3.8, "passed": True, "duration_seconds": 60.0},
        ],
        "summary": {"total": 2, "passed": 2, "failed": 0},
    }

    card = generate_model_card(
        hub_repo="user/my-model",
        model_dir=tmp_path,
        merge_info={"base_model": "org/base-7b"},
        training_config=None,
        eval_results=eval_results,
        language="tr",
    )

    assert "Evaluation Results" in card
    assert "perplexity" in card
    assert "generation" in card
    assert "2/2 benchmarks passed" in card
    assert "PASS" in card


def test_load_merge_info(tmp_path: Path) -> None:
    """_load_merge_info reads merge_info.json when present."""
    from publish_to_hub import _load_merge_info  # noqa: E402

    info = {"base_model": "test/model", "merged": True}
    (tmp_path / "merge_info.json").write_text(json.dumps(info))

    loaded = _load_merge_info(tmp_path)
    assert loaded["base_model"] == "test/model"
    assert loaded["merged"] is True


def test_load_merge_info_missing(tmp_path: Path) -> None:
    """_load_merge_info returns empty dict when file is absent."""
    from publish_to_hub import _load_merge_info  # noqa: E402

    loaded = _load_merge_info(tmp_path)
    assert loaded == {}
