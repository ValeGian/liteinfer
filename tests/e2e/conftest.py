"""E2E test configuration.

vLLM 0.20.0 forks worker processes by default on Linux.  CUDA cannot be
re-initialised inside a forked child when the parent already called into
CUDA (which pytest does via the root conftest's torch.cuda.is_available()).
Setting VLLM_WORKER_MULTIPROC_METHOD=spawn here — before any fixture that
calls LLM() — avoids the RuntimeError.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
