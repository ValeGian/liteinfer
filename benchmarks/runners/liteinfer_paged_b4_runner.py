"""liteinfer with paged KV cache and static batching B=4."""

from __future__ import annotations

from benchmarks.runners.liteinfer_paged_runner import LiteInferPagedCacheRunner


class LiteInferPagedB4Runner(LiteInferPagedCacheRunner):
    name = "liteinfer-paged-b4"
    batch_size = 4

    def setup(self, model: str, **kwargs) -> None:
        kwargs.setdefault("max_num_seqs", 4)
        super().setup(model, **kwargs)
