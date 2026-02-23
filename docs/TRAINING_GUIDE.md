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
FORGE_ALLOW_LOCAL=1 forge train --config configs/models/my_model.yaml --dry-run
```

## Step-by-Step: Fine-Tune for a New Language

### 1. Create Data Config

Copy the template and fill in your language's HuggingFace datasets:

```bash
cp configs/data/template.yaml configs/data/swahili.yaml
# Edit: set language code, dataset repos, output paths
```

Existing language configs: `turkish.yaml`, `azerbaijani.yaml`.

### 2. Download and Preprocess Data

```bash
forge download --config configs/data/swahili.yaml
```

The preprocessor automatically:
- Normalizes formats (alpaca, sharegpt, raw_text, dpo)
- Removes duplicates via MinHash
- Filters by target language (if `target_language` is set)
- Removes short records

### 3. (Optional) Transcribe Audio Data

If you have audio recordings in the target language:

```bash
forge transcribe --audio-dir data/audio/swahili --language sw
```

This generates JSONL files that can be fed into the preprocessing pipeline.

### 4. Create Model Config

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

### 5. Train

```bash
# Verify setup
forge train --config configs/models/my_model.yaml --dry-run

# Train
forge train --config configs/models/my_model.yaml
```

Early stopping is built in via `EarlyStoppingOnPlateau` (configurable patience and min-delta).

### 6. Merge and Evaluate

```bash
forge merge \
    --base-model your-org/your-base-model \
    --adapter artifacts/training/my-model-sft-v1/final \
    --output artifacts/merged/my-model-v1

forge evaluate --model artifacts/merged/my-model-v1
```

### 7. (Optional) Publish to HuggingFace Hub

```bash
forge publish \
    --model-dir artifacts/merged/my-model-v1 \
    --hub-repo your-org/my-model-v1 \
    --training-config configs/models/my_model.yaml \
    --eval-results artifacts/eval/results.json
```

This auto-generates a model card with training config and evaluation results.

## V100-Specific Notes

- Always use `fp16: true` and `bf16: false` in configs
- For 9B models: reduce `max_seq_length` to 1024, `batch_size` to 1
- Enable gradient accumulation (8-16 steps) to compensate for small batch sizes
- Monitor VRAM with `nvidia-smi` during training

## Notebooks

For interactive training analysis:

- `notebooks/02_training_analysis.ipynb` — loss curves, LR schedules, LoRA weight analysis, WandB integration

## Common Issues

**Blocked local run**: command requires remote context; run from SSH session or use `FORGE_ALLOW_LOCAL=1` intentionally.

**Serving returns 401 Unauthorized**: endpoint is API-key protected. Set `FORGE_SERVE_API_KEY` for smoke checks, or pass `--api-key`.

**Out of memory**: Reduce `max_seq_length`, `per_device_train_batch_size`, or `lora_r`.

**Unsloth not found**: Install training extras with `uv sync --extra train`. Pipeline falls back to standard PEFT automatically when Unsloth is unavailable.

**Poor Turkish output**: Check tokenizer coverage — models not trained on Turkish may tokenize inefficiently, reducing effective context length.
