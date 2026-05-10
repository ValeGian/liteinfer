"""Run benchmarks against `liteinfer` with paged KV cache enabled."""

from __future__ import annotations

from benchmarks.runners.liteinfer_runner import LiteInferRunner


class LiteInferPagedCacheRunner(LiteInferRunner):
    name = "liteinfer-paged-kvcache"

    def setup(self, model: str, **kwargs) -> None:
        kwargs.setdefault("cache_mode", "paged")
        super().setup(model, **kwargs)
