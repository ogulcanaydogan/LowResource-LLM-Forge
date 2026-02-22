# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LowResource-LLM-Forge                        │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│  Data       │  Training    │  Evaluation  │  Serving           │
│  Pipeline   │  Pipeline    │  Pipeline    │  Pipeline          │
├─────────────┼──────────────┼──────────────┼────────────────────┤
│ collector   │ trainer      │ evaluator    │ vllm_server        │
│ preprocessor│ merge        │ mmlu_turkish │ replicate_predict  │
│ dataset_    │ callbacks    │ perplexity   │                    │
│   builder   │              │ generation_  │                    │
│             │              │   quality    │                    │
│             │              │ report       │                    │
├─────────────┴──────────────┴──────────────┴────────────────────┤
│  Utils: config (Pydantic + YAML), logging (structlog)          │
├────────────────────────────────────────────────────────────────┤
│  Configs: base.yaml ──► model configs ──► data configs          │
└────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
HuggingFace Datasets
        │
        ▼
   DataCollector ──► data/raw/<lang>/*.jsonl
        │                (normalized alpaca format)
        ▼
  DataPreprocessor ──► data/raw/<lang>/*.clean.jsonl
        │                (cleaned, deduped, filtered)
        ▼
   DatasetBuilder ──► data/processed/<lang>_sft.jsonl
                      data/processed/<lang>_eval.jsonl
                            │
                            ▼
                     ForgeTrainer (QLoRA)
                            │
                            ▼
                   artifacts/training/<run>/final/
                            │
                            ▼
                      LoRAMerger
                            │
                            ▼
                   artifacts/merged/<model>/
                        │           │
                        ▼           ▼
                   ForgeEvaluator  VLLMServer
                        │
                        ▼
                  artifacts/eval/report.md
```

## Config Inheritance

Model-specific YAML configs inherit from `base.yaml` via `_base` key:

```
configs/base.yaml          (shared defaults: fp16, QLoRA, LoRA params)
    ├── configs/models/turkcell_7b.yaml    (overrides: model name, LR)
    ├── configs/models/wiroai_9b.yaml      (overrides: seq_len, batch_size)
    └── configs/models/llama3_8b_tr.yaml   (overrides: model name)
```

## V100 Constraints

The pipeline is optimized for NVIDIA V100 (Volta, compute capability 7.0):

- **fp16 only** — V100 does not support bfloat16
- **4-bit QLoRA** — enables 7B-9B models on 32GB VRAM
- **Gradient checkpointing** — reduces memory at slight speed cost
- **Unsloth optimizations** — 2-5x speedup when available, standard PEFT fallback
