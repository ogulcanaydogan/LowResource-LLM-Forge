# LowResource-LLM-Forge

**Sovereign fine-tuning pipeline that takes low-resource languages from raw data to production inference — data collection, audio transcription, QLoRA training, evaluation, and serving — with V100-first GPU optimization and multi-language support out of the box.**

[![CI](https://github.com/ogulcanaydogan/LowResource-LLM-Forge/actions/workflows/ci.yml/badge.svg)](https://github.com/ogulcanaydogan/LowResource-LLM-Forge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Unsloth](https://img.shields.io/badge/Training-Unsloth%20QLoRA-FF6F00.svg)](https://github.com/unslothai/unsloth)
[![vLLM](https://img.shields.io/badge/Serving-vLLM-7B68EE.svg)](https://github.com/vllm-project/vllm)
[![Tests](https://img.shields.io/badge/Tests-78%20passed-2ECC71.svg)]()
[![mypy](https://img.shields.io/badge/mypy-strict-blue.svg)]()

---

## Why This Exists

Large language models perform well on high-resource languages (English, Chinese, German) but degrade significantly on low-resource languages — Turkish, Azerbaijani, Swahili, Kurdish — where pretraining data is scarce and evaluation benchmarks don't exist. Existing fine-tuning tools assume English data, unlimited GPU budgets, and established evaluation infrastructure.

This creates a structural barrier:

- **No training pipelines** designed for languages with fragmented, low-quality datasets scattered across HuggingFace, audio archives, and scraped corpora.
- **No hardware-aware optimization** for teams working with V100s instead of A100/H100 clusters — most QLoRA tools default to bf16, which V100s don't support.
- **No evaluation methodology** for languages without standardized benchmarks — you can't measure what you can't test.
- **No end-to-end path** from raw multilingual data to a deployed, benchmarked, published model that others can use.

LowResource-LLM-Forge solves this by providing a **single pipeline** that handles the entire lifecycle: collect and clean multilingual data, transcribe audio, fine-tune with V100-optimized QLoRA, evaluate with language-specific benchmarks, serve via vLLM, and publish to HuggingFace Hub — all from one CLI.

The project targets NLP researchers, language preservation teams, and engineering groups building sovereign AI capabilities for underserved language communities.

---

## How It Works

```mermaid
graph LR
    A["Data Sources"] --> B["Data Pipeline"]
    B --> C["QLoRA Training"]
    C --> D["Evaluation"]
    D --> E["Inference Serving"]
    D --> F["HF Hub Publish"]

    style A fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style B fill:#7B68EE,stroke:#5B48CE,color:#fff
    style C fill:#FF6B6B,stroke:#CC4444,color:#fff
    style D fill:#F5A623,stroke:#C7851A,color:#fff
    style E fill:#50C878,stroke:#3AA862,color:#fff
    style F fill:#4ECDC4,stroke:#36B5AC,color:#fff
```

Upload datasets or audio in any supported language — the pipeline normalizes formats, removes duplicates, filters by language, trains a QLoRA adapter on V100-compatible settings, evaluates against configurable benchmarks, and deploys the merged model for inference.

---

## Architecture

```mermaid
graph TD
    subgraph DATA["Data Pipeline"]
        HF["HuggingFace\nDatasets"] --> COLLECT["DataCollector\n(alpaca, sharegpt,\nraw_text, dpo)"]
        AUDIO["Audio Files"] --> WHISPER["WhisperTranscriber\n(language forcing,\nconfidence filter)"]
        WHISPER --> COLLECT
        COLLECT --> PREPROC["DataPreprocessor\n(clean, MinHash dedup,\nlang-filter)"]
        PREPROC --> BUILD["DatasetBuilder\n(train/eval split)"]
    end

    subgraph TRAIN["Training Pipeline"]
        BUILD --> TRAINER["ForgeTrainer\n(Unsloth QLoRA,\nPEFT fallback)"]
        TRAINER --> MERGE["LoRAMerger\n(merge + tokenizer patch)"]
    end

    subgraph EVAL["Evaluation Pipeline"]
        MERGE --> EVALUATOR["ForgeEvaluator"]
        EVALUATOR --> MMLU["Turkish MMLU\n(lm-eval)"]
        EVALUATOR --> PPL["Perplexity\n(held-out text)"]
        EVALUATOR --> GEN["Generation Quality\n(heuristic scoring)"]
    end

    subgraph SERVE["Serving & Publishing"]
        MERGE --> VLLM["vLLM Server\n(systemd, Docker)"]
        MERGE --> REPL["Replicate\n(Cog)"]
        MERGE --> PUB["HF Hub Publish\n(auto model card)"]
    end

    style DATA fill:#1a1a2e,stroke:#4A90D9,color:#fff
    style TRAIN fill:#1a1a2e,stroke:#FF6B6B,color:#fff
    style EVAL fill:#1a1a2e,stroke:#F5A623,color:#fff
    style SERVE fill:#1a1a2e,stroke:#50C878,color:#fff
    style HF fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style AUDIO fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style COLLECT fill:#7B68EE,stroke:#5B48CE,color:#fff
    style WHISPER fill:#7B68EE,stroke:#5B48CE,color:#fff
    style PREPROC fill:#7B68EE,stroke:#5B48CE,color:#fff
    style BUILD fill:#7B68EE,stroke:#5B48CE,color:#fff
    style TRAINER fill:#FF6B6B,stroke:#CC4444,color:#fff
    style MERGE fill:#FF8C42,stroke:#CC6A2E,color:#fff
    style EVALUATOR fill:#F5A623,stroke:#C7851A,color:#fff
    style MMLU fill:#F5A623,stroke:#C7851A,color:#fff
    style PPL fill:#F5A623,stroke:#C7851A,color:#fff
    style GEN fill:#F5A623,stroke:#C7851A,color:#fff
    style VLLM fill:#50C878,stroke:#3AA862,color:#fff
    style REPL fill:#50C878,stroke:#3AA862,color:#fff
    style PUB fill:#4ECDC4,stroke:#36B5AC,color:#fff
```

---

## Key Capabilities

### Multi-Format Data Pipeline

```mermaid
graph LR
    A["HuggingFace\nDatasets"] --> N["Format\nNormalizer"]
    B["Audio\nRecordings"] --> W["Whisper\nTranscriber"]
    W --> N
    N --> C["Text\nCleaner"]
    C --> D["MinHash\nDedup"]
    D --> E{"Language\nFilter"}
    E -->|Pass| F["SFT Dataset\n+ Eval Split"]
    E -->|Reject| G["Filtered Out"]

    style A fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style B fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style W fill:#7B68EE,stroke:#5B48CE,color:#fff
    style N fill:#7B68EE,stroke:#5B48CE,color:#fff
    style C fill:#7B68EE,stroke:#5B48CE,color:#fff
    style D fill:#7B68EE,stroke:#5B48CE,color:#fff
    style E fill:#F5A623,stroke:#C7851A,color:#fff
    style F fill:#50C878,stroke:#3AA862,color:#fff
    style G fill:#E74C3C,stroke:#C0392B,color:#fff
```

| Feature | Description |
|---------|-------------|
| **Format normalization** | Alpaca, ShareGPT, raw text, DPO — all converted to unified SFT format |
| **Whisper transcription** | Audio-to-text with language forcing and log-probability confidence filtering |
| **MinHash deduplication** | Near-duplicate removal across large corpora |
| **Language detection** | Keyword-overlap heuristic for Turkic languages (Turkish, Azerbaijani) with extensible marker sets |
| **Quality filtering** | Minimum length enforcement, character validation, configurable thresholds |

### V100-Optimized QLoRA Training

```mermaid
graph TD
    CFG["YAML Config\n(inherits base.yaml)"] --> LOAD["Load Base Model\n4-bit NF4 Quantization"]
    LOAD --> LORA["Attach LoRA Adapters\n(r=32, α=64)"]
    LORA --> OPT{"Unsloth\nAvailable?"}
    OPT -->|Yes| FAST["Unsloth Optimized\n(2-5x speedup)"]
    OPT -->|No| STD["Standard PEFT\n(Fallback)"]
    FAST --> TRAIN["Train\n(fp16, gradient checkpoint,\nearly stopping)"]
    STD --> TRAIN
    TRAIN --> SAVE["Save Adapter\n+ Merge to Base"]

    style CFG fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style LOAD fill:#7B68EE,stroke:#5B48CE,color:#fff
    style LORA fill:#7B68EE,stroke:#5B48CE,color:#fff
    style OPT fill:#F5A623,stroke:#C7851A,color:#fff
    style FAST fill:#50C878,stroke:#3AA862,color:#fff
    style STD fill:#FF8C42,stroke:#CC6A2E,color:#fff
    style TRAIN fill:#FF6B6B,stroke:#CC4444,color:#fff
    style SAVE fill:#4ECDC4,stroke:#36B5AC,color:#fff
```

| Constraint | Design Decision |
|-----------|-----------------|
| **fp16 only** | V100 (Volta) does not support bf16 — all configs enforce `fp16: true, bf16: false` |
| **4-bit QLoRA** | NF4 quantization enables 7B-9B models on 16-32GB VRAM |
| **Gradient checkpointing** | Trades compute for memory — critical for 9B models on 32GB V100 |
| **Unsloth fast path** | 2-5x training speedup when available, transparent PEFT fallback when not |
| **Early stopping** | `EarlyStoppingOnPlateau` with configurable patience and min-delta |
| **Config inheritance** | `base.yaml` → model-specific YAML via `_base` key, deep merge semantics |

### Evaluation Framework

| Benchmark | Method | Pass Threshold | Scope |
|-----------|--------|---------------|-------|
| **Turkish MMLU** | lm-evaluation-harness (`turkishmmlu` task) | ≥0.40 accuracy | Broad academic knowledge |
| **Perplexity** | Cross-entropy on held-out eval set | <50.0 | Language modeling quality |
| **Generation Quality** | Heuristic scoring on 10 diverse Turkish prompts | ≥3.5/5.0 | Fluency, coherence, character usage |

### Remote-First Execution Safety

```mermaid
graph TD
    CMD["forge train / evaluate / serve / publish"] --> GUARD{"Runtime\nGuard"}
    GUARD -->|SSH Session| ALLOW["Execute"]
    GUARD -->|CI Job| ALLOW
    GUARD -->|FORGE_EXECUTION_CONTEXT=remote| ALLOW
    GUARD -->|Local Shell| BLOCK["Block with\nRuntimeError"]
    BLOCK -->|FORGE_ALLOW_LOCAL=1| ALLOW

    style CMD fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style GUARD fill:#F5A623,stroke:#C7851A,color:#fff
    style ALLOW fill:#50C878,stroke:#3AA862,color:#fff
    style BLOCK fill:#E74C3C,stroke:#C0392B,color:#fff
```

Model-loading commands are blocked on local development machines by default. Only SSH sessions, CI jobs, and explicitly marked remote contexts are allowed — preventing accidental multi-hour GPU processes on laptops.

### Deployment Options

```mermaid
graph LR
    MODEL["Merged Model"] --> VLLM["vLLM\n(systemd, SSH deploy)"]
    MODEL --> DOCKER["Docker Compose\n(train + serve)"]
    MODEL --> COG["Replicate\n(Cog)"]
    MODEL --> HUB["HF Hub\n(auto model card)"]

    VLLM --> API["OpenAI-compatible\n/v1/completions"]
    DOCKER --> API
    COG --> RAPI["Replicate API"]

    style MODEL fill:#FF8C42,stroke:#CC6A2E,color:#fff
    style VLLM fill:#50C878,stroke:#3AA862,color:#fff
    style DOCKER fill:#50C878,stroke:#3AA862,color:#fff
    style COG fill:#50C878,stroke:#3AA862,color:#fff
    style HUB fill:#4ECDC4,stroke:#36B5AC,color:#fff
    style API fill:#7B68EE,stroke:#5B48CE,color:#fff
    style RAPI fill:#7B68EE,stroke:#5B48CE,color:#fff
```

| Target | Method | Features |
|--------|--------|----------|
| **vLLM** | SSH + user-level systemd | Versioned model dirs, atomic symlink switching, API key enforcement, DGX Spark eager mode |
| **Docker** | `docker compose` | Training + serving containers, GPU passthrough |
| **Replicate** | Cog build + push | Managed inference API |
| **HuggingFace Hub** | Auto model card generation | Training config, eval results, usage examples embedded in card |

---

## Multi-Language Support

```mermaid
graph TD
    TEMPLATE["configs/data/template.yaml"] --> TR["Turkish\n(primary)"]
    TEMPLATE --> AZ["Azerbaijani"]
    TEMPLATE --> NEW["Your Language\n(copy template)"]

    TR --> MARKERS["Language Markers\n(keyword detection)"]
    AZ --> MARKERS
    NEW --> MARKERS

    style TEMPLATE fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style TR fill:#50C878,stroke:#3AA862,color:#fff
    style AZ fill:#50C878,stroke:#3AA862,color:#fff
    style NEW fill:#F5A623,stroke:#C7851A,color:#fff
    style MARKERS fill:#7B68EE,stroke:#5B48CE,color:#fff
```

| Language | Config | Datasets | Status |
|----------|--------|----------|--------|
| **Turkish** | `configs/data/turkish.yaml` | Turkce-Instruct-Merged, turkish-text-data | Primary |
| **Azerbaijani** | `configs/data/azerbaijani.yaml` | AzInstruct_merged | Configured |
| **New language** | Copy `configs/data/template.yaml` | Bring your own datasets | Template ready |

Adding a new language requires:
1. A data config YAML with HuggingFace dataset sources
2. Optionally, language marker words in `preprocessor.py` for detection filtering
3. A model config inheriting from `configs/base.yaml`

---

## Supported Models

| Model | Base Architecture | V100 Compatible | Config |
|-------|-------------------|-----------------|--------|
| **Turkcell-LLM-7b-v1** | Mistral | Yes (primary target) | `configs/models/turkcell_7b.yaml` |
| **wiroai-turkish-llm-9b** | Gemma | Yes (tight on 32GB) | `configs/models/wiroai_9b.yaml` |
| **cere-llama-3-8b-tr** | Llama 3 | Yes | `configs/models/llama3_8b_tr.yaml` |

---

## Quick Start

```bash
# 1. Install
uv sync --extra dev

# 2. Download and preprocess Turkish data
forge download --config configs/data/turkish.yaml

# 3. Fine-tune on remote GPU host (SSH session)
forge train --config configs/models/turkcell_7b.yaml

# 4. Merge adapters into base model
forge merge --base-model TURKCELL/Turkcell-LLM-7b-v1 \
    --adapter artifacts/training/turkcell-7b-sft-v1/final \
    --output artifacts/merged/turkcell-7b-turkish-v1

# 5. Evaluate
forge evaluate --model artifacts/merged/turkcell-7b-turkish-v1

# 6. Publish to HuggingFace Hub
forge publish --model-dir artifacts/merged/turkcell-7b-turkish-v1 \
    --hub-repo ogulcanaydogan/turkcell-7b-turkish-sft \
    --training-config configs/models/turkcell_7b.yaml
```

## CLI Reference

All pipeline stages are accessible via the `forge` CLI:

| Command | Purpose |
|---------|---------|
| `forge download --config ...` | Download and preprocess training data |
| `forge transcribe --audio-dir ...` | Transcribe audio files via Whisper |
| `forge train --config ...` | Run QLoRA fine-tuning |
| `forge evaluate --model ...` | Run evaluation benchmarks |
| `forge merge --base-model ... --adapter ... --output ...` | Merge LoRA adapters into base model |
| `forge serve --config ...` | Start vLLM inference server |
| `forge publish --model-dir ... --hub-repo ...` | Publish model to HuggingFace Hub |
| `forge benchmark --base-url ...` | Benchmark an OpenAI-compatible endpoint |

All commands support `--help` for full option documentation. Run `make help` to see all Makefile targets.

---

## Post-Completion Roadmap

After the current priority training run is completed, the next improvement work is tracked in:

- `docs/ROADMAP.md`

Roadmap phases:

1. Stability hardening (NaN guards, fail-fast, auto-resume)
2. Turkish data expansion and quality filtering
3. A100 training recipe optimization
4. Serving throughput and latency optimization
5. Evaluation depth and release governance

---

## Notebooks

Interactive Jupyter notebooks for exploration and analysis:

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_data_exploration.ipynb` | Dataset statistics, language distribution, preprocessing quality, sample inspection |
| `notebooks/02_training_analysis.ipynb` | Loss curves, learning rate schedules, LoRA weight analysis, WandB integration |

---

## Development

```bash
uv sync --extra dev
make help          # Show all available targets
make qa            # Run all quality gates (lint + typecheck + test)
make test          # pytest
make lint          # ruff
make typecheck     # mypy
```

### Quality Gate Status

| Check | Tool | Status |
|-------|------|--------|
| Lint | ruff | 0 issues |
| Types | mypy (strict) | 0 issues in 25 source files |
| Tests | pytest | 78 passed |
| Coverage | pytest-cov | 47% (threshold: 40%) |

---

## Hardware Requirements

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Training (7B QLoRA) | V100 16GB | V100 32GB |
| Training (9B QLoRA) | V100 32GB | A100 40GB |
| Inference | 16GB VRAM | DGX Spark (GB10) |
| Data Pipeline | CPU only | 16GB RAM |
| Audio Transcription | CPU (slow) | GPU with 8GB VRAM |

---

## License

Apache-2.0
