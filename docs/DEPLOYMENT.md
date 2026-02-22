# Deployment Guide

## DGX Spark (vLLM)

### Prerequisites
- SSH access to DGX Spark
- Merged model at `artifacts/merged/<model-name>/`
- User-level systemd available on DGX (`systemctl --user`)

### Deploy

```bash
bash scripts/deploy_vllm.sh deploy artifacts/merged/turkcell-7b-turkish-v1 configs/serving/vllm_dgx.yaml
```

This will:
1. Create deployment directory on Spark
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
journalctl --user -u forge-vllm.service
```

### Verify

```bash
curl http://spark-5fc3:8000/health
curl http://spark-5fc3:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "model", "prompt": "Merhaba, nasılsın?", "max_tokens": 100}'
```

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
# API available at http://localhost:8000/v1
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
