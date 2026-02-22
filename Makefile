.PHONY: dev test lint typecheck train eval serve smoke-serve download-data ops-dashboard

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

smoke-serve:
	@if [ -z "$(SERVE_BASE_URL)" ]; then \
		echo "Usage: make smoke-serve SERVE_BASE_URL=http://<host>:<port> [EXPECT_MODEL=<id>]"; \
		exit 1; \
	fi
	@if [ -n "$(EXPECT_MODEL)" ]; then \
		uv run python scripts/smoke_serve.py --base-url "$(SERVE_BASE_URL)" --expected-model "$(EXPECT_MODEL)"; \
	else \
		uv run python scripts/smoke_serve.py --base-url "$(SERVE_BASE_URL)"; \
	fi

download-data:
	uv run python scripts/download_data.py --config $(DATA_CONFIG)

ops-dashboard:
	bash scripts/runtime_dashboard.sh
