# Training Guide

## Prerequisites

- NVIDIA GPU with ≥16GB VRAM (V100 32GB recommended)
- CUDA 12.1+
- Python 3.11+
- Run training/eval/merge from a remote host (SSH session) by default

## Runtime Safety Guard

Model-loading commands are blocked in local shells by default. Allowed contexts:

- SSH sessions on DGX/V100 hosts
- CI jobs
- `FORGE_EXECUTION_CONTEXT=remote`

Intentional local override:

```bash
FORGE_ALLOW_LOCAL=1 uv run python scripts/run_training.py --config configs/models/my_model.yaml --dry-run
```

## Step-by-Step: Fine-Tune for a New Language

### 1. Create Data Config

Copy the template and fill in your language's HuggingFace datasets:

```bash
cp configs/data/template.yaml configs/data/swahili.yaml
# Edit: set language code, dataset repos, output paths
```

### 2. Download and Preprocess Data

```bash
uv run python scripts/download_data.py --config configs/data/swahili.yaml
```

### 3. Create Model Config

```yaml
# configs/models/my_model.yaml
_base: "../base.yaml"

model:
  name: "your-org/your-base-model"
  max_seq_length: 2048

data:
  train_path: "data/processed/swahili_sft.jsonl"
  eval_path: "data/processed/swahili_eval.jsonl"

wandb:
  run_name: "my-model-sft-v1"
```

### 4. Train

```bash
# Verify setup
uv run python scripts/run_training.py --config configs/models/my_model.yaml --dry-run

# Train
uv run python scripts/run_training.py --config configs/models/my_model.yaml
```

### 5. Merge and Evaluate

```bash
uv run python scripts/merge_and_push.py \
    --base-model your-org/your-base-model \
    --adapter artifacts/training/my-model-sft-v1/final \
    --output artifacts/merged/my-model-v1

uv run python scripts/run_eval.py --model artifacts/merged/my-model-v1
```

## V100-Specific Notes

- Always use `fp16: true` and `bf16: false` in configs
- For 9B models: reduce `max_seq_length` to 1024, `batch_size` to 1
- Enable gradient accumulation (8-16 steps) to compensate for small batch sizes
- Monitor VRAM with `nvidia-smi` during training

## Common Issues

**Blocked local run**: command requires remote context; run from SSH session or use `FORGE_ALLOW_LOCAL=1` intentionally.

**Out of memory**: Reduce `max_seq_length`, `per_device_train_batch_size`, or `lora_r`.

**Unsloth not found**: Install with `uv sync --extra unsloth`. Falls back to standard PEFT automatically.

**Poor Turkish output**: Check tokenizer coverage — models not trained on Turkish may tokenize inefficiently, reducing effective context length.
