"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from forge.utils.config import (
    EvalConfig,
    LoRAConfig,
    ServingConfig,
    TrainingParams,
    _deep_merge,
    load_data_config,
    load_serving_config,
    load_training_config,
    load_yaml_config,
)


def test_deep_merge_simple() -> None:
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested() -> None:
    base = {"model": {"name": "test", "seq_len": 1024}, "lr": 0.01}
    override = {"model": {"seq_len": 2048}, "epochs": 3}
    result = _deep_merge(base, override)
    assert result["model"]["name"] == "test"
    assert result["model"]["seq_len"] == 2048
    assert result["epochs"] == 3


def test_lora_config_defaults() -> None:
    config = LoRAConfig()
    assert config.r == 32
    assert config.alpha == 64
    assert config.dropout == 0.05
    assert "q_proj" in config.target_modules
    assert len(config.target_modules) == 7


def test_training_params_v100_defaults() -> None:
    params = TrainingParams()
    assert params.fp16 is True
    assert params.bf16 is False  # V100 does NOT support bf16


def test_load_yaml_config(tmp_path: Path) -> None:
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml.dump({"model": {"name": "test-model"}, "seed": 42}))
    result = load_yaml_config(config_file)
    assert result["model"]["name"] == "test-model"
    assert result["seed"] == 42


def test_load_yaml_config_with_base(tmp_path: Path) -> None:
    base_file = tmp_path / "base.yaml"
    base_file.write_text(yaml.dump({
        "seed": 42,
        "training": {"fp16": True, "bf16": False, "learning_rate": 0.001},
        "lora": {"r": 32, "alpha": 64},
    }))

    child_file = tmp_path / "child.yaml"
    child_file.write_text(yaml.dump({
        "_base": "base.yaml",
        "training": {"learning_rate": 0.0002},
        "model": {"name": "test/model"},
    }))

    result = load_yaml_config(child_file)
    assert result["seed"] == 42
    assert result["training"]["fp16"] is True
    assert result["training"]["learning_rate"] == 0.0002
    assert result["model"]["name"] == "test/model"
    assert result["lora"]["r"] == 32


def test_load_training_config(tmp_path: Path) -> None:
    config_file = tmp_path / "model.yaml"
    config_data = {
        "model": {"name": "test/model", "max_seq_length": 2048, "dtype": "float16"},
        "training": {"num_epochs": 3, "learning_rate": 0.0002},
        "data": {
            "train_path": "data/train.jsonl",
            "eval_path": "data/eval.jsonl",
        },
    }
    config_file.write_text(yaml.dump(config_data))

    cfg = load_training_config(config_file)
    assert cfg.model.name == "test/model"
    assert cfg.training.num_epochs == 3
    assert cfg.train_data_path == "data/train.jsonl"
    assert cfg.training.fp16 is True


def test_load_data_config(tmp_path: Path) -> None:
    config_file = tmp_path / "data.yaml"
    config_data = {
        "language": "tr",
        "language_name": "Turkish",
        "sources": {
            "huggingface": [
                {"repo": "test/dataset", "split": "train", "format": "alpaca"},
            ],
        },
        "preprocessing": {"min_length": 100},
        "output": {"sft_path": "data/sft.jsonl", "eval_split_ratio": 0.1},
    }
    config_file.write_text(yaml.dump(config_data))

    cfg = load_data_config(config_file)
    assert cfg.language == "tr"
    assert cfg.language_name == "Turkish"
    assert len(cfg.sources) == 1
    assert cfg.sources[0].repo == "test/dataset"
    assert cfg.preprocessing.min_length == 100
    assert cfg.output.eval_split_ratio == 0.1


def test_eval_config_defaults() -> None:
    cfg = EvalConfig(model_path="test/model")
    assert "perplexity" in cfg.benchmarks
    assert "generation" in cfg.benchmarks
    assert "mmlu_tr" not in cfg.benchmarks
    assert cfg.device == "cuda"


def test_serving_config_defaults() -> None:
    cfg = ServingConfig(model_path="test/model")
    assert cfg.port == 8000
    assert cfg.dtype == "float16"
    assert cfg.gpu_memory_utilization == 0.90
    assert cfg.trust_remote_code is False
    assert cfg.enforce_eager is False


def test_load_serving_config_enforce_eager(tmp_path: Path) -> None:
    config_file = tmp_path / "serving.yaml"
    config_file.write_text(
        yaml.dump(
            {
                "model_path": "/tmp/model",
                "port": 18000,
                "enforce_eager": True,
            }
        )
    )

    cfg = load_serving_config(config_file)
    assert cfg.model_path == "/tmp/model"
    assert cfg.port == 18000
    assert cfg.enforce_eager is True
