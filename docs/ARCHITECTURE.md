# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        LowResource-LLM-Forge                            │
├──────────────┬──────────────┬──────────────┬─────────────┬─────────────┤
│  Data        │  Training    │  Evaluation  │  Serving    │  Publishing │
│  Pipeline    │  Pipeline    │  Pipeline    │  Pipeline   │  Pipeline   │
├──────────────┼──────────────┼──────────────┼─────────────┼─────────────┤
│ collector    │ trainer      │ evaluator    │ vllm_server │ publish_to_ │
│ preprocessor │ merge        │ mmlu_turkish │ replicate_  │   hub.py    │
│ dataset_     │ callbacks    │ perplexity   │   predict   │             │
│   builder    │              │ generation_  │             │             │
│ whisper_     │              │   quality    │             │             │
│  transcriber │              │ report       │             │             │
├──────────────┴──────────────┴──────────────┴─────────────┴─────────────┤
│  Utils: config (Pydantic + YAML), logging (structlog), runtime_guard   │
├──────────────────────────────────────────────────────────────────────────┤
│  Configs: base.yaml ──► model configs ──► data configs ──► serving     │
└──────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
HuggingFace Datasets         Audio Files
        │                        │
        ▼                        ▼
   DataCollector          WhisperTranscriber
        │                   (language forcing,
        │                    confidence filter)
        │                        │
        ▼                        ▼
  DataPreprocessor ◄──── transcriptions.jsonl
        │  (clean, dedup,
        │   lang-filter)
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
                    │           │           │
                    ▼           ▼           ▼
              ForgeEvaluator  VLLMServer  publish_to_hub
                    │                       │
                    ▼                       ▼
              artifacts/eval/        HuggingFace Hub
              report.md              (model + card)
```

## Config Inheritance

Model-specific YAML configs inherit from `base.yaml` via `_base` key:

```
configs/base.yaml          (shared defaults: fp16, QLoRA, LoRA params)
    ├── configs/models/turkcell_7b.yaml    (overrides: model name, LR)
    ├── configs/models/wiroai_9b.yaml      (overrides: seq_len, batch_size)
    └── configs/models/llama3_8b_tr.yaml   (overrides: model name)

configs/data/template.yaml (data config template)
    ├── configs/data/turkish.yaml
    └── configs/data/azerbaijani.yaml
```

## Multi-Language Support

Language detection uses a keyword-overlap heuristic (`detect_language_heuristic`) with marker sets for Turkic languages. The preprocessor filters records that don't match the target language when `target_language` is set in config.

Currently supported: Turkish (tr), Azerbaijani (az). New languages can be added by providing marker word sets in `preprocessor.py`.

## V100 Constraints

The pipeline is optimized for NVIDIA V100 (Volta, compute capability 7.0):

- **fp16 only** — V100 does not support bfloat16
- **4-bit QLoRA** — enables 7B-9B models on 32GB VRAM
- **Gradient checkpointing** — reduces memory at slight speed cost
- **Unsloth optimizations** — 2-5x speedup when available, standard PEFT fallback

## CLI Architecture

The `forge` CLI (Click-based) provides unified access to all pipeline stages:

```
forge download   → DataCollector + DataPreprocessor + DatasetBuilder
forge train      → ForgeTrainer
forge evaluate   → ForgeEvaluator
forge merge      → LoRAMerger
forge transcribe → WhisperTranscriber
forge publish    → scripts/publish_to_hub.py (subprocess)
forge serve      → VLLMServer
forge benchmark  → scripts/benchmark_openai_endpoint.py (subprocess)
```

GPU-dependent commands use lazy imports to avoid loading torch/transformers on dev machines. Remote-first safety guard blocks model-loading operations in local shells by default.
