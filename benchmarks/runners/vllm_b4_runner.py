"""vLLM with static batching B=4."""

from __future__ import annotations

from benchmarks.runners.vllm_runner import VLLMRunner


class VLLMB4Runner(VLLMRunner):
    """vLLM with `max_num_seqs=4` for direct comparison against `liteinfer-b4`."""

    name = "vllm-b4"
    batch_size = 4

    def setup(self, model: str, **kwargs) -> None:
        kwargs.setdefault("max_num_seqs", 4)
        super().setup(model, **kwargs)
