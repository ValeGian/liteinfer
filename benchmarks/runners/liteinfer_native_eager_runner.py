"""Run benchmarks against `liteinfer` with native eager KV cache enabled."""

from __future__ import annotations

from benchmarks.runners.liteinfer_runner import LiteInferRunner


class LiteInferNativeEagerCacheRunner(LiteInferRunner):
    """liteinfer with cache_mode='native_eager' (plain-tensor KV cache, no DynamicCache).

    Decode steps use the native KV cache: only the new token is passed to the
    model each step. Contrast with 'liteinfer-kvcache' which wraps transformers'
    DynamicCache, and with the default RECOMPUTE runner which re-feeds the full
    growing sequence on every step.
    """

    name = "liteinfer-native-kvcache"

    def setup(self, model: str, **kwargs) -> None:
        kwargs.setdefault("cache_mode", "native_eager")
        super().setup(model, **kwargs)
