#!/usr/bin/env bash
# Set up Bonsai Chat: virtual environment, app dependencies, the model pack in
# a models directory, and the model pack's pinned MLX runtime.
# Safe to re-run; finished steps and already-downloaded files are skipped.
set -euo pipefail
cd -- "$(dirname -- "$0")"

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

  --models-dir DIR     Models directory. The pack is placed in its own folder,
                       DIR/Ternary-Bonsai-2-27B-mlx-2bit.
                       Default: $BONSAI_MODELS_DIR, otherwise ~/models
  --move-from DIR      Move an already downloaded pack into the models
                       directory instead of downloading about 8.6 GB again.
  --download           Download a new copy even if a pack exists at the old
                       ~/projects/ternary-bonsai-2/bonsai2-27b-mlx location.
  --no-verify          Skip SHA-256 verification of the downloaded files.
  --skip-runtime-deps  Do not install the pack's runtime/requirements.txt.
  --mlx-backend NAME   Linux only: cpu (default), cuda12 or cuda13.
  --python PATH        Python used to create .venv (default: python3).
  -h, --help           Show this help.
EOF
}

PYTHON_BIN="${PYTHON:-python3}"
MLX_BACKEND="${BONSAI_MLX_BACKEND:-cpu}"
SKIP_RUNTIME_DEPS=0
MODELS_DIR=""
MODEL_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --models-dir)
      [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 64; }
      MODELS_DIR="$2"; MODEL_ARGS+=("$1" "$2"); shift 2 ;;
    --move-from)
      [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 64; }
      MODEL_ARGS+=("$1" "$2"); shift 2 ;;
    --download|--no-verify) MODEL_ARGS+=("$1"); shift ;;
    --skip-runtime-deps) SKIP_RUNTIME_DEPS=1; shift ;;
    --mlx-backend)
      [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 64; }
      MLX_BACKEND="$2"; shift 2 ;;
    --python)
      [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 64; }
      PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 64 ;;
  esac
done

case "$MLX_BACKEND" in
  cpu|cuda12|cuda13) ;;
  *) echo "--mlx-backend must be cpu, cuda12 or cuda13" >&2; exit 64 ;;
esac

step() { printf '\n==> %s\n' "$1"; }

OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" = "Darwin" ] && [ "$ARCH" != "arm64" ]; then
  echo "Warning: MLX needs Apple Silicon; this Mac reports $ARCH." >&2
fi

step "Python environment (.venv)"
if [ ! -x .venv/bin/python ]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Python not found: $PYTHON_BIN. Install Python 3.10+ or pass --python." >&2; exit 1; }
  "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "$PYTHON_BIN is older than Python 3.10, which MLX requires." >&2
    echo "Install a newer Python (for example 'brew install python') and pass --python." >&2
    exit 1; }
  "$PYTHON_BIN" -m venv .venv || {
    echo "Could not create .venv. On Debian/Ubuntu: sudo apt install python3-venv" >&2; exit 1; }
fi
VENV_PY=".venv/bin/python"
"$VENV_PY" -m pip install --quiet --upgrade pip

step "App dependencies"
"$VENV_PY" -m pip install --quiet -r requirements.txt huggingface_hub

step "Model pack"
# ${arr[@]+...} keeps an empty array safe under 'set -u' on macOS's bash 3.2.
"$VENV_PY" install_model.py ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}

if [ -n "$MODELS_DIR" ]; then
  MODEL_DIR="$("$VENV_PY" install_model.py --print-path --models-dir "$MODELS_DIR")"
else
  MODEL_DIR="$("$VENV_PY" install_model.py --print-path)"
fi

if [ "$SKIP_RUNTIME_DEPS" -eq 0 ]; then
  step "Model runtime dependencies (pinned by the model pack)"
  RUNTIME_REQ="$MODEL_DIR/runtime/requirements.txt"
  if [ "$OS" = "Linux" ]; then
    # On Linux the 'mlx' wheel has no compute backend unless an extra is chosen.
    MLX_PIN="$(grep -E '^mlx==' "$RUNTIME_REQ" | head -n 1 | tr -d '[:space:]')"
    [ -n "$MLX_PIN" ] || { echo "No 'mlx==' pin found in $RUNTIME_REQ" >&2; exit 1; }
    "$VENV_PY" -m pip install --quiet -r "$RUNTIME_REQ" "mlx[$MLX_BACKEND]==${MLX_PIN#mlx==}"
  else
    "$VENV_PY" -m pip install --quiet -r "$RUNTIME_REQ"
  fi
  "$VENV_PY" -c 'import mlx.core as mx; print("MLX", mx.__version__, "on", mx.default_device())'
fi

step "Done"
echo "Model: $MODEL_DIR"
echo "Start the app with: ./run.sh"
