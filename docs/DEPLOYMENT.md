# Deployment Guide

## vLLM on Remote GPU Hosts (user-level systemd)

### Prerequisites
- SSH access to target host (key or password auth)
- Merged model at `artifacts/merged/<model-name>/`
- User-level systemd available (`systemctl --user`)
- Remote Python with `vllm` installed (see `python_bin` in serving config)

### Deploy

```bash
# Default target (spark via SSH config)
bash scripts/deploy_vllm.sh deploy artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_dgx.yaml

# Override target host/user
DEPLOY_HOST=10.34.9.233 DEPLOY_USER=weezboo bash scripts/deploy_vllm.sh deploy \
  artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_dgx.yaml

# Password SSH fallback
DEPLOY_HOST=10.34.9.233 DEPLOY_USER=weezboo SSH_PASSWORD='***' bash scripts/deploy_vllm.sh deploy \
  artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_dgx.yaml
```

This will:
1. Create deployment directory on remote host
2. Sync model weights via rsync
3. Install/update `~/.config/systemd/user/forge-vllm.service`
4. Reload systemd user daemon
5. Enable and restart `forge-vllm.service`

### Service Management

```bash
bash scripts/deploy_vllm.sh status
bash scripts/deploy_vllm.sh start
bash scripts/deploy_vllm.sh stop
bash scripts/deploy_vllm.sh restart
bash scripts/deploy_vllm.sh logs
```

Logs are read from:

```bash
journalctl --user -u forge-vllm
```

### Verify

```bash
# Warmup can take ~30-90s on first start.
curl http://<host>:18000/health
curl http://<host>:18000/v1/models
curl http://<host>:18000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "model", "prompt": "Merhaba, nasılsın?", "max_tokens": 100}'

# Project smoke check helper
make smoke-serve SERVE_BASE_URL=http://<host>:18000
```

## CI Smoke Check

`ci.yml` can run the same serving smoke check automatically when these repository secrets are set:

- `FORGE_SERVE_BASE_URL` (required)
- `FORGE_SERVE_EXPECT_MODEL` (optional)

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

### Test

```bash
curl -s -X POST https://api.replicate.com/v1/predictions \
    -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
    -d '{"version": "<version-id>", "input": {"prompt": "Merhaba"}}'
```
