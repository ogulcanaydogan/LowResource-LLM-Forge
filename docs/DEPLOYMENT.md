# Deployment Guide

## vLLM on Remote GPU Hosts (user-level systemd)

### Prerequisites
- SSH access to target host (key or password auth)
- Merged model at `artifacts/merged/<model-name>/`
- User-level systemd available (`systemctl --user`)
- Remote Python with `vllm` installed (see `python_bin` in serving config)

### Deploy

```bash
export VLLM_API_KEY='forge-change-this-to-a-long-random-secret'

# Spark (GB10)
VLLM_API_KEY="$VLLM_API_KEY" bash scripts/deploy_vllm.sh deploy \
  artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_spark.yaml

# Override target host/user (VM330 V100 profile)
DEPLOY_HOST=10.34.9.233 DEPLOY_USER=weezboo VLLM_API_KEY="$VLLM_API_KEY" \
  bash scripts/deploy_vllm.sh deploy \
  artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_dgx.yaml

# Password SSH fallback
DEPLOY_HOST=10.34.9.233 DEPLOY_USER=weezboo SSH_PASSWORD='***' \
  VLLM_API_KEY="$VLLM_API_KEY" bash scripts/deploy_vllm.sh deploy \
  artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_dgx.yaml
```

Remote bind (`host: 0.0.0.0`) requires `VLLM_API_KEY` (or `FORGE_SERVE_API_KEY`).
Bypass only for controlled internal debugging: `ALLOW_INSECURE_SERVE=1`.

This will:
1. Create deployment directory on remote host
2. Sync model weights to `~/llm-forge/models/<model_name>/` via rsync
3. Update `~/llm-forge/model-active` symlink to the synced model
4. Install/update `~/.config/systemd/user/forge-vllm.service`
5. Reload systemd user daemon
6. Enable and restart `forge-vllm.service`

`configs/serving/vllm_spark.yaml` sets `enforce_eager: true` to avoid Triton/PTX compile failures on GB10 GPUs.

### Service Management

```bash
bash scripts/deploy_vllm.sh status
bash scripts/deploy_vllm.sh start
bash scripts/deploy_vllm.sh stop
bash scripts/deploy_vllm.sh restart
bash scripts/deploy_vllm.sh logs
bash scripts/deploy_vllm.sh set-active <model_name_under_~/llm-forge/models>
bash scripts/deploy_vllm.sh reconfigure configs/serving/vllm_spark.yaml
```

Logs: `journalctl --user -u forge-vllm`

### Start via CLI

```bash
forge serve --config configs/serving/vllm_dgx.yaml
```

### Verify

```bash
# Warmup can take ~30-90s on first start.
curl http://<host>:18000/health
curl http://<host>:18000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"

# Project smoke check helper
make smoke-serve SERVE_BASE_URL=http://<host>:18000 SERVE_API_KEY="$VLLM_API_KEY"

# Endpoint benchmarking
forge benchmark --base-url http://<host>:18000/v1 --api-key "$VLLM_API_KEY"
```

### Ops Dashboard

Monitor running services:

```bash
make ops-dashboard
```

## CI Smoke Check

- `ci.yml` includes an optional smoke step and skips on GitHub-hosted runners.
- `eval-gate.yml` (self-hosted GPU runner) can execute smoke checks with `workflow_dispatch` inputs:
  - `serve_base_url` (optional, enables smoke check when set)
  - `serve_expected_model` (optional)
  - `serve_api_key` (optional, required for protected endpoints)

## Docker

### Training

```bash
# Run from remote GPU host
docker compose -f deploy/docker-compose.yml up train
```

### Serving

```bash
# Mount your merged model
docker compose -f deploy/docker-compose.yml up serve
# API available at http://localhost:18000/v1
```

## Replicate

### Prerequisites
- Install Cog: `pip install cog`
- Replicate account and API token

### Deploy

1. Copy merged model weights to `model_weights/` directory
2. Build and push:

```bash
cd deploy
cog build
cog push r8.im/ogulcanaydogan/turkcell-7b-turkish-sft
```

## Publishing to HuggingFace Hub

Publish a merged model with an auto-generated model card:

```bash
forge publish \
    --model-dir artifacts/merged/turkcell-7b-turkish-v1 \
    --hub-repo ogulcanaydogan/turkcell-7b-turkish-sft \
    --training-config configs/models/turkcell_7b.yaml \
    --eval-results artifacts/eval/results.json
```

The model card includes training configuration, evaluation results, usage examples, and dataset provenance.

## Environment Variables

See `.env.example` for all available environment variables. Key deployment variables:

| Variable | Purpose |
|----------|---------|
| `VLLM_API_KEY` | API key for vLLM endpoint |
| `FORGE_SERVE_API_KEY` | Alias for VLLM_API_KEY |
| `DEPLOY_HOST` | Target SSH host |
| `DEPLOY_USER` | Target SSH user |
| `SSH_PASSWORD` | SSH password (prefer key auth) |
| `ALLOW_INSECURE_SERVE` | Allow 0.0.0.0 without API key |
| `HF_TOKEN` | HuggingFace token for publishing |
