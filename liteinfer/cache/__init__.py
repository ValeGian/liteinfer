"""KV cache for continuous batching.

`ContinuousKVCache` stores each sequence's tokens in fixed-size blocks drawn
from a shared `BlockPool`. Block addressing lives in `slot_table`, so reads and
writes are single indexing ops rather than per-sequence Python loops.
"""

from liteinfer.cache.block_pool import BlockPool, slot_table
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache

__all__ = ["BlockPool", "ContinuousKVCache", "slot_table"]
