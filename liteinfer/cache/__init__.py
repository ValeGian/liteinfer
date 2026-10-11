"""KV cache for continuous batching.

`KVCache` stores each sequence's tokens in fixed-size blocks drawn from a
shared `BlockPool`. Block addressing lives in `packed_addresses` (where a
packed pass writes and what it reads), `newest_slots` (where a decode step
writes) and `slot_table` (left-aligned rows of every token a sequence holds), so
reads and writes are single indexing ops rather than per-sequence Python loops.
"""

from liteinfer.cache.block_pool import BlockPool, newest_slots, packed_addresses, slot_table
from liteinfer.cache.kv_cache import KVCache

__all__ = ["BlockPool", "KVCache", "newest_slots", "packed_addresses", "slot_table"]
