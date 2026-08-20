"""KV cache implementations for the inference engine.

Every cache implements the unified :class:`KVCache` interface from
:mod:`inference.kv_cache.base`:

* :class:`NaiveKVCache`   -- ``torch.cat`` per update (reference implementation)
* :class:`StaticKVCache`  -- one pre-allocated buffer, in-place writes
* :class:`DynamicKVCache` -- capacity grows by doubling on demand
* :class:`PagedKVCache`   -- fixed-size physical blocks + per-sequence block tables
"""

from inference.kv_cache.base import KVCache
from inference.kv_cache.block import Block
from inference.kv_cache.block_manager import BlockManager
from inference.kv_cache.dynamic import DynamicKVCache
from inference.kv_cache.naive import NaiveKVCache
from inference.kv_cache.paged import PagedKVCache, SequenceState
from inference.kv_cache.static import StaticKVCache

__all__ = [
    "KVCache",
    "Block",
    "BlockManager",
    "NaiveKVCache",
    "StaticKVCache",
    "DynamicKVCache",
    "PagedKVCache",
    "SequenceState",
]
