#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export BONSAI_MODEL_PATH="${BONSAI_MODEL_PATH:-$HOME/projects/ternary-bonsai-2/bonsai2-27b-mlx}"
exec python3 app.py
