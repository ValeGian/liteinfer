"""Run benchmarks against `liteinfer` with eager KV cache enabled."""

from __future__ import annotations

from benchmarks.runners.liteinfer_runner import LiteInferRunner


class LiteInferEagerCacheRunner(LiteInferRunner):
    """liteinfer with cache_mode='eager' (DynamicCache via transformers).

    Decode steps use the KV cache: only the new token is passed to the model
    each step. Prefill still runs the full prompt once to populate the cache.
    Contrast with the default RECOMPUTE runner, which re-feeds the full and
    growing sequence on every step.
    """

    name = "liteinfer-kvcache"

    def setup(self, model: str, **kwargs) -> None:
        kwargs.setdefault("cache_mode", "eager")
        super().setup(model, **kwargs)
