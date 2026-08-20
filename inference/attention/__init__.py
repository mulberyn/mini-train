"""Inference attention modules.

* :func:`dense_attention` / :func:`attention_with_positions` -- reference
  attention with position-aware causal masking.
* :class:`KVAttention` -- inference MHA that reads/writes a KV cache.
* :func:`paged_attention` / :func:`paged_attention_from_cache` -- paged
  attention over a block-table based KV pool.
"""

from inference.attention.attention import (
    attention_with_positions,
    dense_attention,
    kv_causal_mask,
)
from inference.attention.kv_attention import KVAttention
from inference.attention.paged_attention import paged_attention, paged_attention_from_cache

__all__ = [
    "kv_causal_mask",
    "dense_attention",
    "attention_with_positions",
    "KVAttention",
    "paged_attention",
    "paged_attention_from_cache",
]
