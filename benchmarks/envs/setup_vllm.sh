#!/usr/bin/env bash
# Creates the vLLM venv and installs vllm==0.21.0.
# Requires: CUDA, Python 3.10+, sufficient VRAM (8 GB for 1B model, 24 GB for 8B).
set -euo pipefail

VENV_DIR="$(dirname "$0")/vllm"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install "vllm==0.21.0"

echo "vLLM venv created at: $VENV_DIR"
echo "Python: $VENV_DIR/bin/python"
