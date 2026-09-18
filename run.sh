#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# app.py finds the model itself: BONSAI_MODEL_PATH, then the location recorded
# by ./install.sh, then ~/models/Ternary-Bonsai-2-27B-mlx-2bit.
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python app.py
fi
exec python3 app.py
