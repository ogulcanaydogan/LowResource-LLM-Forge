.PHONY: help dev test lint typecheck qa train eval serve smoke-serve download-data transcribe publish benchmark ops-dashboard

.DEFAULT_GOAL := help

TRAIN_CONFIG ?= configs/models/turkcell_7b.yaml
SERVE_CONFIG ?= configs/serving/vllm_dgx.yaml
DATA_CONFIG ?= configs/data/turkish.yaml

help: ## Show all available targets
	@echo "LowResource-LLM-Forge — Make targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

dev: ## Install dev dependencies
	uv sync --extra dev

test: ## Run pytest suite
	uv run pytest

lint: ## Run ruff linter
	uv run ruff check .

typecheck: ## Run mypy type checker
	uv run mypy src/forge

qa: lint typecheck test ## Run all quality gates (lint + typecheck + test)

train: ## Fine-tune model (TRAIN_CONFIG=<yaml>)
	uv run python scripts/run_training.py --config $(TRAIN_CONFIG)

eval: ## Evaluate model (MODEL=<path>)
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make eval MODEL=<model_path_or_repo>"; \
		exit 1; \
	fi
	uv run python scripts/run_eval.py --model "$(MODEL)"

serve: ## Start vLLM server (SERVE_CONFIG=<yaml>)
	uv run python scripts/run_serve.py --config $(SERVE_CONFIG)

smoke-serve: ## Health-check a running endpoint (SERVE_BASE_URL=<url>)
	@if [ -z "$(SERVE_BASE_URL)" ]; then \
		echo "Usage: make smoke-serve SERVE_BASE_URL=http://<host>:<port> [EXPECT_MODEL=<id>] [SERVE_API_KEY=<key>]"; \
		exit 1; \
	fi
	@cmd='uv run python scripts/smoke_serve.py --base-url "$(SERVE_BASE_URL)"'; \
	if [ -n "$(EXPECT_MODEL)" ]; then cmd="$$cmd --expected-model \"$(EXPECT_MODEL)\""; fi; \
	if [ -n "$(SERVE_API_KEY)" ]; then cmd="$$cmd --api-key \"$(SERVE_API_KEY)\""; fi; \
	eval "$$cmd"

download-data: ## Download and preprocess data (DATA_CONFIG=<yaml>)
	uv run python scripts/download_data.py --config $(DATA_CONFIG)

transcribe: ## Transcribe audio with Whisper (AUDIO_DIR=<path>)
	@if [ -z "$(AUDIO_DIR)" ]; then \
		echo "Usage: make transcribe AUDIO_DIR=<path> [WHISPER_CONFIG=configs/data/whisper.yaml]"; \
		exit 1; \
	fi
	uv run python scripts/transcribe_audio.py --audio-dir "$(AUDIO_DIR)" \
		$(if $(WHISPER_CONFIG),--config "$(WHISPER_CONFIG)",)

publish: ## Publish model to HF Hub (MODEL_DIR=<path> HUB_REPO=<repo>)
	@if [ -z "$(MODEL_DIR)" ] || [ -z "$(HUB_REPO)" ]; then \
		echo "Usage: make publish MODEL_DIR=<merged_model_path> HUB_REPO=<user/repo> [TRAIN_CONFIG=<yaml>] [EVAL_RESULTS=<json>]"; \
		exit 1; \
	fi
	@cmd='uv run python scripts/publish_to_hub.py --model-dir "$(MODEL_DIR)" --hub-repo "$(HUB_REPO)"'; \
	if [ -n "$(TRAIN_CONFIG)" ]; then cmd="$$cmd --training-config \"$(TRAIN_CONFIG)\""; fi; \
	if [ -n "$(EVAL_RESULTS)" ]; then cmd="$$cmd --eval-results \"$(EVAL_RESULTS)\""; fi; \
	eval "$$cmd"

benchmark: ## Benchmark endpoint (BASE_URL=<url> [API_KEY=<key>])
	@if [ -z "$(BASE_URL)" ]; then \
		echo "Usage: make benchmark BASE_URL=http://<host>:<port> [API_KEY=<key>] [NUM_REQUESTS=50] [CONCURRENCY=5]"; \
		exit 1; \
	fi
	@cmd='uv run python scripts/benchmark_openai_endpoint.py --base-url "$(BASE_URL)"'; \
	if [ -n "$(API_KEY)" ]; then cmd="$$cmd --api-key \"$(API_KEY)\""; fi; \
	if [ -n "$(NUM_REQUESTS)" ]; then cmd="$$cmd --num-requests $(NUM_REQUESTS)"; fi; \
	if [ -n "$(CONCURRENCY)" ]; then cmd="$$cmd --concurrency $(CONCURRENCY)"; fi; \
	eval "$$cmd"

ops-dashboard: ## Launch runtime ops dashboard
	bash scripts/runtime_dashboard.sh
