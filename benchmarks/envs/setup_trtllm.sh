#!/usr/bin/env bash
# Creates the TensorRT-LLM venv and installs tensorrt-llm==1.2.1.
# Requires: CUDA >= 12.1, Python 3.10+, >= 24 GB VRAM for 8B model runs.
# Uses the PyTorch backend — no trtllm-build compilation step required.
set -euo pipefail

VENV_DIR="$(dirname "$0")/trtllm"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install "tensorrt-llm==1.2.1" \
  --extra-index-url https://pypi.nvidia.com

echo "TRT-LLM venv created at: $VENV_DIR"
echo "Python: $VENV_DIR/bin/python"
