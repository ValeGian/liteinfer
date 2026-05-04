"""Eager per-sequence KV cache.

Wraps `transformers.cache_utils.DynamicCache`. The point of going
through liteinfer's `KVCache` ABC rather than passing `DynamicCache`
around directly is to keep the engine code untouched when paged or
prefix-shared variants are introduced.
"""

from __future__ import annotations

from transformers.cache_utils import DynamicCache  # type: ignore[attr-defined]

from liteinfer.cache.kv_cache import KVCache
from liteinfer.config import EngineConfig


class EagerKVCache(KVCache):
    def __init__(self, config: EngineConfig, hf_config) -> None:
        super().__init__(config)
        self._hf_config = hf_config
        self._inner = DynamicCache(config=hf_config)

    def reset(self) -> None:
        self._inner = DynamicCache(config=self._hf_config)

    def get_seq_length(self) -> int:
        return self._inner.get_seq_length()

    @property
    def payload(self) -> DynamicCache:
        return self._inner
