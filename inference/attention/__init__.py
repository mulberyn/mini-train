"""Inference attention modules.

* :func:`dense_attention` / :func:`attention_with_positions` -- reference
  attention with position-aware causal masking.
* :class:`KVAttention` -- inference MHA that reads/writes a KV cache.
* :func:`paged_attention` / :func:`paged_attention_from_cache` -- paged
  attention over a block-table based KV pool, with pluggable
  implementations: ``"loop"`` (reference), ``"torch"`` (batch-vectorized,
  default) and ``"triton"`` (native kernel).
"""

from inference.attention.attention import (
    attention_with_positions,
    dense_attention,
    kv_causal_mask,
)
from inference.attention.kv_attention import KVAttention
from inference.attention.paged_attention import (
    IMPLEMENTATIONS,
    paged_attention,
    paged_attention_from_cache,
)
from inference.attention.triton.paged_attention import (
    TRITON_AVAILABLE,
    paged_attention_triton,
)

__all__ = [
    "kv_causal_mask",
    "dense_attention",
    "attention_with_positions",
    "KVAttention",
    "paged_attention",
    "paged_attention_from_cache",
    "IMPLEMENTATIONS",
    "paged_attention_triton",
    "TRITON_AVAILABLE",
]
