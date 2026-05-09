"""liteinfer with static batching B=4 + eager KV cache."""

from __future__ import annotations

from benchmarks.runners.liteinfer_runner import LiteInferRunner


class LiteInferB4Runner(LiteInferRunner):
    """liteinfer with `max_num_seqs=4` and `cache_mode="eager"`.

    Exercises the static-batching path (roadmap §1.1). Up to 4 sequences
    share one PREFILL and decode together; eager KV cache keeps decode
    cost O(1) per step per sequence.
    """

    name = "liteinfer-b4"
    batch_size = 4

    def setup(self, model: str, **kwargs) -> None:
        kwargs.setdefault("cache_mode", "eager")
        kwargs.setdefault("max_num_seqs", 4)
        super().setup(model, **kwargs)
