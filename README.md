# LowResource-LLM-Forge

Sovereign LLM fine-tuning pipeline for low-resource languages. End-to-end system for data collection, QLoRA fine-tuning, evaluation, and inference serving.

## Architecture

```
Audio/Text Sources ──► Data Pipeline ──► QLoRA Training ──► Evaluation ──► Inference
                       (collect,          (Unsloth,        (MMLU-TR,       (vLLM,
                        clean,             PEFT,            perplexity,     Replicate)
                        dedup)             V100)            generation)
```

## Quick Start

```bash
# 1. Install
uv sync --extra dev

# 2. Download and preprocess Turkish data
uv run python scripts/download_data.py --config configs/data/turkish.yaml

# 3. Fine-tune on remote GPU host (SSH session)
uv run python scripts/run_training.py --config configs/models/turkcell_7b.yaml
```

## Remote-First Safety

Model-loading commands are blocked on local shells by default:

- `scripts/run_training.py`
- `scripts/run_eval.py`
- `scripts/run_serve.py`
- `scripts/merge_and_push.py`

Allowed by default in:

- SSH sessions (`SSH_CONNECTION` detected)
- CI jobs (`CI=true`)
- Explicit remote context (`FORGE_EXECUTION_CONTEXT=remote`)

To intentionally bypass on local machine:

```bash
FORGE_ALLOW_LOCAL=1 uv run python scripts/run_eval.py --model TURKCELL/Turkcell-LLM-7b-v1
```

## Pipeline Stages

### Data Pipeline

Download datasets from HuggingFace, clean, deduplicate, and build training splits:

```bash
uv run python scripts/download_data.py --config configs/data/turkish.yaml --limit 1000
```

Supports formats: alpaca, sharegpt, raw_text, dpo. Add new languages by copying `configs/data/template.yaml`.

### Training (QLoRA on V100)

Fine-tune with 4-bit quantization using Unsloth (falls back to standard PEFT if unavailable):

```bash
# Dry run — verify model loads
uv run python scripts/run_training.py --config configs/models/turkcell_7b.yaml --dry-run

# Full training
uv run python scripts/run_training.py --config configs/models/turkcell_7b.yaml

# Quick test (100 steps)
uv run python scripts/run_training.py --config configs/models/turkcell_7b.yaml --max-steps 100
```

### Merge & Push

Merge LoRA adapters into base model for efficient inference:

```bash
uv run python scripts/merge_and_push.py \
    --base-model TURKCELL/Turkcell-LLM-7b-v1 \
    --adapter artifacts/training/turkcell-7b-sft-v1/final \
    --output artifacts/merged/turkcell-7b-turkish-v1 \
    --push --hub-repo ogulcanaydogan/turkcell-7b-turkish-sft
```

### Evaluation

Run benchmarks against base or fine-tuned models:

```bash
# All benchmarks
uv run python scripts/run_eval.py --model artifacts/merged/turkcell-7b-turkish-v1

# Single benchmark
uv run python scripts/run_eval.py --model TURKCELL/Turkcell-LLM-7b-v1 --benchmark perplexity
```

| Benchmark | Method | Pass Threshold |
|-----------|--------|---------------|
| Turkish MMLU | lm-evaluation-harness | ≥0.40 accuracy |
| Perplexity | Held-out Turkish text | <50.0 |
| Generation | Heuristic scoring on Turkish prompts | ≥3.5/5.0 |

### Inference Serving

**vLLM (remote host, user-level systemd):**

```bash
# Key-based SSH host (default: spark)
bash scripts/deploy_vllm.sh deploy artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_dgx.yaml

# Password SSH host example (VM330)
DEPLOY_HOST=10.34.9.233 DEPLOY_USER=weezboo SSH_PASSWORD='***' \
  bash scripts/deploy_vllm.sh deploy artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_dgx.yaml

bash scripts/deploy_vllm.sh status
bash scripts/deploy_vllm.sh logs

# Start service process directly on a remote host
uv run python scripts/run_serve.py --config configs/serving/vllm_dgx.yaml
```

First startup may take ~30-90 seconds due model load and graph warmup. Wait for `/health` to return `200`.

**Docker:**

```bash
docker compose -f deploy/docker-compose.yml up serve
```

## Supported Models

| Model | Base | V100 Compatible | Config |
|-------|------|-----------------|--------|
| Turkcell-LLM-7b-v1 | Mistral | Yes (primary) | `configs/models/turkcell_7b.yaml` |
| wiroai-turkish-llm-9b | Gemma | Yes (tight) | `configs/models/wiroai_9b.yaml` |
| cere-llama-3-8b-tr | Llama3 | Yes | `configs/models/llama3_8b_tr.yaml` |

## Adding a New Language

1. Copy `configs/data/template.yaml` to `configs/data/<language>.yaml`
2. Fill in HuggingFace dataset sources
3. Run the data pipeline: `uv run python scripts/download_data.py --config configs/data/<language>.yaml`
4. Create a model config inheriting from `configs/base.yaml`
5. Train and evaluate

## Development

```bash
uv sync --extra dev
make test          # Run tests
make lint          # Run ruff
make typecheck     # Run mypy
make eval MODEL=TURKCELL/Turkcell-LLM-7b-v1
```

## Hardware Requirements

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Training (7B QLoRA) | V100 16GB | V100 32GB |
| Training (9B QLoRA) | V100 32GB | A100 40GB |
| Inference | 16GB VRAM | DGX Spark |
| Data Pipeline | CPU only | 16GB RAM |

## License

Apache-2.0
