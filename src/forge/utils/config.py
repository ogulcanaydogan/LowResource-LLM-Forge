"""Pydantic config models loaded from YAML with _base inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    r: int = 32        # rank 32 is sweet spot for 7B, 16 enough for smaller
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"  # V100 has no bf16
    bnb_4bit_use_double_quant: bool = True


class TrainingParams(BaseModel):
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    eval_steps: int = 100
    fp16: bool = True   # always True for Volta arch
    bf16: bool = False  # NOT supported on V100
    max_steps: int = -1
    seed: int = 42


class ModelConfig(BaseModel):
    name: str
    max_seq_length: int = 2048  # 4096 OOMs on V100 32GB with batch>1
    dtype: str = "float16"


class WandbConfig(BaseModel):
    project: str = "lowresource-llm-forge"
    run_name: str | None = None
    enabled: bool = True


class TrainingConfig(BaseModel):
    """Top-level training config. Defaults tuned for V100."""

    model: ModelConfig
    training: TrainingParams = Field(default_factory=TrainingParams)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    train_data_path: str = ""
    eval_data_path: str = ""
    output_dir: str = "artifacts/training"


class DataSourceConfig(BaseModel):
    repo: str
    split: str = "train"
    format: str = "alpaca"
    subset: str | None = None


class PreprocessingConfig(BaseModel):
    min_length: int = 50
    max_length: int = 8192
    dedup_method: str = "minhash"
    dedup_threshold: float = 0.85  # empirically tuned on turkish instruction data
    clean_html: bool = True
    normalize_unicode: bool = True
    target_language: str | None = None


class DataOutputConfig(BaseModel):
    sft_path: str = "data/processed/sft.jsonl"
    dpo_path: str = "data/processed/dpo.jsonl"
    eval_path: str = "data/processed/eval.jsonl"
    eval_split_ratio: float = 0.05


class DataConfig(BaseModel):
    """Data pipeline config loaded from configs/data/*.yaml."""

    language: str
    language_name: str
    sources: list[DataSourceConfig] = Field(default_factory=list)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    output: DataOutputConfig = Field(default_factory=DataOutputConfig)


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    model_path: str
    base_model_name: str = ""
    benchmarks: list[str] = Field(default_factory=lambda: ["perplexity", "generation"])
    output_dir: str = "artifacts/eval"
    device: str = "cuda"


class ServingConfig(BaseModel):
    """vLLM serving configuration."""

    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90  # leave headroom for KV cache
    max_model_len: int = 4096
    dtype: str = "float16"
    enable_prefix_caching: bool = True  # ~15% latency gain on repeated system prompts
    max_num_seqs: int = 64
    trust_remote_code: bool = False
    enforce_eager: bool = False
    api_key: str | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge override into base, recursing into nested dicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config with optional _base inheritance."""
    config_path = Path(config_path)
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}  # empty YAML returns None

    base_ref = data.pop("_base", None)
    if base_ref:
        base_path = (config_path.parent / base_ref).resolve()
        base_data = load_yaml_config(base_path)
        data = _deep_merge(base_data, data)

    return data


def load_training_config(config_path: str | Path) -> TrainingConfig:
    """Load and validate a training configuration from YAML."""
    raw = load_yaml_config(config_path)

    if "data" in raw:
        data_section = raw.pop("data")
        raw.setdefault("train_data_path", data_section.get("train_path", ""))
        raw.setdefault("eval_data_path", data_section.get("eval_path", ""))

    return TrainingConfig(**raw)


def load_data_config(config_path: str | Path) -> DataConfig:
    """Load and validate a data pipeline configuration from YAML."""
    raw = load_yaml_config(config_path)

    hf_sources = raw.get("sources", {}).get("huggingface", [])
    sources = [DataSourceConfig(**s) for s in hf_sources]

    return DataConfig(
        language=raw["language"],
        language_name=raw["language_name"],
        sources=sources,
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        output=DataOutputConfig(**raw.get("output", {})),
    )


def load_eval_config(config_path: str | Path) -> EvalConfig:
    """Load and validate an evaluation configuration from YAML."""
    raw = load_yaml_config(config_path)
    return EvalConfig(**raw)


def load_serving_config(config_path: str | Path) -> ServingConfig:
    """Load and validate a serving configuration from YAML."""
    raw = load_yaml_config(config_path)
    return ServingConfig(**raw)
