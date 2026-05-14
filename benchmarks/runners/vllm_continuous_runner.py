"""Benchmark runner for vLLM with continuous batching enabled.

vLLM always runs continuous batching internally. This runner exposes it with
``max_num_seqs=32`` (versus the existing ``vllm`` runner which forces B=1) so
the harness can compare vLLM's full continuous-batching throughput against
liteinfer's async engine.
"""

from __future__ import annotations

from benchmarks.runners.vllm_runner import VLLMRunner


class VLLMContinuousRunner(VLLMRunner):
    name = "vllm-continuous"
    batch_size = 4

    def setup(self, model: str, **kwargs) -> None:
        kwargs.setdefault("max_num_seqs", 4)
        super().setup(model, **kwargs)
