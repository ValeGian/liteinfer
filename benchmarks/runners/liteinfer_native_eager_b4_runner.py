"""liteinfer with native eager KV cache and static batching B=4."""

from __future__ import annotations

from benchmarks.runners.liteinfer_native_eager_runner import LiteInferNativeEagerCacheRunner


class LiteInferNativeEagerB4Runner(LiteInferNativeEagerCacheRunner):
    name = "liteinfer-native-kvcache-b4"
    batch_size = 4

    def setup(self, model: str, **kwargs) -> None:
        kwargs.setdefault("max_num_seqs", 4)
        super().setup(model, **kwargs)
