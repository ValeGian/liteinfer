#!/usr/bin/env bash
# Pulls the TGI Docker image.
# Requires: Docker with GPU support (nvidia-container-toolkit), CUDA >= 12.1.
set -euo pipefail

docker pull ghcr.io/huggingface/text-generation-inference:latest

echo "TGI image pulled: ghcr.io/huggingface/text-generation-inference:latest"
