#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/LowResource-LLM-Forge}"
cd "$PROJECT_ROOT"

bash scripts/monitor_v100_training.sh
