"""KV cache abstract base.

The engine creates one cache per static batch and threads it through
``model.forward(..., past_key_values=cache.payload, use_cache=True)``.

Different implementations decide layout and reuse policy:

- ``EagerKVCache`` — per-sequence K/V tensors, no sharing. The
  baseline; matches what ``transformers`` does internally.
- ``PagedKVCache`` *(future)* — block-allocated tensors that allow
  packing variable-length sequences and continuous batching.
- ``PrefixKVCache`` *(future)* — paged cache with deduplicated common
  prefixes.

The "no cache" execution mode is **not** a `KVCache` subclass: the
engine simply skips cache instantiation and passes ``use_cache=False``
to the model. Keeping that out of the type hierarchy avoids a
spurious null-object class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from liteinfer.config import EngineConfig


class KVCache(ABC):
    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    @abstractmethod
    def reset(self) -> None:
        """Clear all cached state. Called between static batches."""

    @abstractmethod
    def get_seq_length(self) -> int:
        """Number of tokens currently cached for the (static) batch."""

    @property
    @abstractmethod
    def payload(self) -> Any:
        """Object passed to ``model.forward(past_key_values=…)``.

        Concrete shape is implementation-defined — the engine treats it
        as opaque.
        """
