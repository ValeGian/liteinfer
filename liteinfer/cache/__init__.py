"""KV cache for continuous batching.

Each sequence's tokens live in fixed-size blocks of a shared `BlockPool` (device
storage); `BlockTables` records which blocks each sequence holds and which are
free (host bookkeeping, which the scheduler reads too); `KVCache` turns the two
into what a pass writes and reads. Block addressing lives in `packed_addresses`
(where a packed pass writes and what it reads), `newest_slots` (where a decode
step writes) and `slot_table` (left-aligned rows of every token a sequence
holds), so reads and writes are single indexing ops rather than per-sequence
Python loops.
"""

from liteinfer.cache.block_pool import BlockPool, newest_slots, packed_addresses, slot_table
from liteinfer.cache.block_tables import BlockPoolExhaustedError, BlockTables
from liteinfer.cache.kv_cache import KVCache

__all__ = [
    "BlockPool",
    "BlockPoolExhaustedError",
    "BlockTables",
    "KVCache",
    "newest_slots",
    "packed_addresses",
    "slot_table",
]
