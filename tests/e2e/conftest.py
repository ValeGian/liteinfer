"""E2E test configuration.

vLLM forks worker processes by default on Linux. CUDA cannot be re-initialised
inside a forked child once the parent has called into CUDA (which pytest does
via the root conftest's torch.cuda.is_available()), so force spawn before any
fixture constructs an engine.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
