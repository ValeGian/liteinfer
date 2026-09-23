"""KV cache for continuous batching.

`ContinuousKVCache` stores each sequence's tokens in fixed-size blocks drawn
from a shared `BlockPool`. Block addressing lives in `slot_mapping` (one slot per
token, sequences end to end) and `slot_table` (right-aligned rows), so reads and
writes are single indexing ops rather than per-sequence Python loops.
"""

from liteinfer.cache.block_pool import BlockPool, slot_mapping, slot_table
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache

__all__ = ["BlockPool", "ContinuousKVCache", "slot_mapping", "slot_table"]
