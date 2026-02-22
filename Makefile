.PHONY: dev test lint typecheck train eval serve download-data

TRAIN_CONFIG ?= configs/models/turkcell_7b.yaml
SERVE_CONFIG ?= configs/serving/vllm_dgx.yaml
DATA_CONFIG ?= configs/data/turkish.yaml

dev:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy src/forge

train:
	uv run python scripts/run_training.py --config $(TRAIN_CONFIG)

eval:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make eval MODEL=<model_path_or_repo>"; \
		exit 1; \
	fi
	uv run python scripts/run_eval.py --model "$(MODEL)"

serve:
	uv run python scripts/run_serve.py --config $(SERVE_CONFIG)

download-data:
	uv run python scripts/download_data.py --config $(DATA_CONFIG)
