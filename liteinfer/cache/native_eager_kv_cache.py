# pyright: reportPrivateImportUsage=false
"""Native eager KV cache — stores raw tensors, no transformers.DynamicCache dependency."""

from __future__ import annotations

import torch

from liteinfer.cache.kv_cache import KVCache
from liteinfer.config import EngineConfig


class _NativeCachePayload:
    """Per-layer KV store backed by plain tensors.

    Shape convention: [batch, num_heads, seq_len, head_dim]
    """

    def __init__(self) -> None:
        self._keys: list[torch.Tensor | None] = []
        self._values: list[torch.Tensor | None] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V for *layer_idx* and return the full accumulated K/V."""
        while len(self._keys) <= layer_idx:
            self._keys.append(None)
            self._values.append(None)

        existing_k = self._keys[layer_idx]
        existing_v = self._values[layer_idx]
        if existing_k is None or existing_v is None:
            merged_k = key_states
            merged_v = value_states
        else:
            merged_k = torch.cat([existing_k, key_states], dim=2)
            merged_v = torch.cat([existing_v, value_states], dim=2)

        self._keys[layer_idx] = merged_k
        self._values[layer_idx] = merged_v
        return merged_k, merged_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._keys) or self._keys[layer_idx] is None:
            return 0
        return int(self._keys[layer_idx].shape[2])  # type: ignore[union-attr]

    def clear(self) -> None:
        self._keys = []
        self._values = []


class NativeEagerKVCache(KVCache):
    """Eager KV cache backed by plain tensors — no transformers.DynamicCache wrapper."""

    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self._inner = _NativeCachePayload()

    def reset(self) -> None:
        self._inner.clear()

    def get_seq_length(self) -> int:
        return self._inner.get_seq_length()

    @property
    def payload(self) -> _NativeCachePayload:
        return self._inner
