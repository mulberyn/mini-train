"""Paged attention with pluggable implementations.

Given a batch of single-token queries whose K/V live in a paged block pool,
compute one attention output per query by walking each sequence's *block
table* (logical block -> physical block):

    logical block 0 -> physical block 17
    logical block 1 -> physical block 4
    ...

Implementations (``implementation=...``):

* ``"loop"``   -- per-sequence Python loop over blocks. Correctness-first
  reference; runtime scales as ``O(B)`` because every sequence launches its
  own kernels (the bottleneck documented in ``docs/paged_attention.md``).
* ``"torch"``  -- batch-vectorized: gathers all blocks of all sequences in one
  ``key_cache[block_tables]`` indexing op, then runs batched matmuls. This
  restores GPU parallelism across the batch (runtime ~flat in ``B``).
* ``"triton"`` -- Triton kernel, one program per ``(batch, head)`` with a
  flash-attention style online-softmax accumulator (see
  ``inference/attention/triton/paged_attention.py``). Requires ``triton``
  (Linux + CUDA; not available on this Windows host, tests auto-skip).

The benchmark ``benchmark/attention/benchmark_batch_scaling.py`` compares the
three and shows the loop's O(B) scaling, the vectorized torch path staying
flat, and (where triton runs) the native kernel.
"""

from __future__ import annotations

import math

import torch

from inference.kv_cache.paged import PagedKVCache

IMPLEMENTATIONS = ("loop", "torch", "triton")


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    block_size: int,
    num_kv_heads: int | None = None,
    scale: float | None = None,
    implementation: str = "torch",
) -> torch.Tensor:
    """Paged attention for a batch of single-token queries.

    Args:
        query: ``[B, num_heads, D]`` query tokens (decode step).
        key_cache: ``[num_blocks, block_size, num_kv_heads, D]`` physical pool.
        value_cache: same shape as ``key_cache``.
        block_tables: ``[B, max_num_blocks]`` int64 mapping each sequence's
            logical blocks to physical blocks (``-1`` padding ignored).
        context_lengths: ``[B]`` number of cached tokens per sequence.
        block_size: tokens per block (power of two for ``"triton"``).
        num_kv_heads: KV heads in the pool; defaults to ``query`` heads (MHA).
            For GQA the query heads are grouped (head ``i`` reads KV head
            ``i // (num_heads // num_kv_heads)``).
        scale: score scale; defaults to ``1 / sqrt(D)``.
        implementation: ``"loop"`` | ``"torch"`` (default) | ``"triton"``.

    Returns:
        ``[B, num_heads, D]`` attention outputs.
    """
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(
            f"unknown implementation {implementation!r} "
            f"(expected one of {IMPLEMENTATIONS})"
        )
    if query.ndim != 3:
        raise ValueError(f"query must be [B, H, D], got {tuple(query.shape)}")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must have the same shape")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_tables.ndim != 2:
        raise ValueError(f"block_tables must be [B, max_blocks], got {tuple(block_tables.shape)}")
    if context_lengths.ndim != 1 or context_lengths.numel() != query.shape[0]:
        raise ValueError(
            f"context_lengths must be 1-D with {query.shape[0]} entries"
        )

    batch, num_heads, head_dim = query.shape
    num_kv_heads = num_kv_heads or num_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
    heads_per_kv = num_heads // num_kv_heads
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    block_tables = block_tables.to(query.device)
    context_lengths = context_lengths.to(query.device)
    query = query.to(key_cache.device) if query.device != key_cache.device else query

    if implementation == "loop":
        return _paged_attention_loop(
            query, key_cache, value_cache, block_tables, context_lengths,
            block_size, num_kv_heads, heads_per_kv, scale,
        )
    if implementation == "torch":
        return _paged_attention_vectorized(
            query, key_cache, value_cache, block_tables, context_lengths,
            block_size, num_kv_heads, heads_per_kv, scale,
        )
    # implementation == "triton"
    from inference.attention.triton.paged_attention import paged_attention_triton

    return paged_attention_triton(
        query, key_cache, value_cache, block_tables, context_lengths,
        block_size, num_kv_heads, scale,
    )


def _check_block_tables(block_tables, context_lengths, block_size, key_cache):
    """Shared validation: block-table columns vs required blocks, ids in range."""
    max_ctx = int(context_lengths.max().item())
    max_blocks = (max_ctx + block_size - 1) // block_size
    if max_blocks > block_tables.shape[1]:
        raise IndexError(
            f"sequences need {max_blocks} blocks but the block table has only "
            f"{block_tables.shape[1]} columns"
        )
    tables = block_tables[:, :max_blocks]
    if bool((tables < 0).any()):
        raise RuntimeError(
            "block table contains -1 (unallocated) entries for a non-empty sequence"
        )
    if bool((tables >= key_cache.shape[0]).any()):
        raise IndexError(
            f"block id out of pool range [0, {key_cache.shape[0]})"
        )
    return tables


def _paged_attention_loop(
    query, key_cache, value_cache, block_tables, context_lengths,
    block_size, num_kv_heads, heads_per_kv, scale,
):
    """Per-sequence reference implementation (O(B) kernels)."""
    batch, num_heads, head_dim = query.shape
    outputs = []
    for b in range(batch):
        q = query[b]  # [num_heads, D]
        ctx = int(context_lengths[b])
        if ctx == 0:
            outputs.append(
                torch.zeros(num_heads, head_dim, dtype=query.dtype, device=query.device)
            )
            continue
        tables = _check_block_tables(block_tables, context_lengths, block_size, key_cache)
        num_logical_blocks = (ctx + block_size - 1) // block_size
        k_parts, v_parts = [], []
        for logical in range(num_logical_blocks):
            physical = int(tables[b, logical])
            k = key_cache[physical]  # [block_size, num_kv_heads, D]
            v = value_cache[physical]
            if logical == num_logical_blocks - 1:
                rem = ctx - logical * block_size
                k = k[:rem]
                v = v[:rem]
            k_parts.append(k)
            v_parts.append(v)

        k = torch.cat(k_parts, dim=0)  # [ctx, num_kv_heads, D]
        v = torch.cat(v_parts, dim=0)
        if num_kv_heads != num_heads:
            kv_head_index = torch.arange(num_heads, device=k.device) // heads_per_kv
            k = k[:, kv_head_index, :]
            v = v[:, kv_head_index, :]
        k = k.transpose(0, 1)  # [num_heads, ctx, D]
        v = v.transpose(0, 1)

        scores = torch.einsum("hd,htd->ht", q, k) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("ht,htd->hd", probs, v)
        outputs.append(out)
    return torch.stack(outputs, dim=0)


def _paged_attention_vectorized(
    query, key_cache, value_cache, block_tables, context_lengths,
    block_size, num_kv_heads, heads_per_kv, scale,
):
    """Batch-vectorized implementation: one gather + batched matmuls.

    All sequences' K/V blocks are gathered in a single ``key_cache[block_tables]``
    advanced-indexing op and every head/sequence is processed by the same
    batched kernels, so runtime stays ~flat as the batch grows (unlike the
    loop implementation's ``O(B)`` kernel launches).
    """
    batch, num_heads, head_dim = query.shape
    ctxs = context_lengths
    max_ctx = int(ctxs.max().item())
    if max_ctx == 0:
        return torch.zeros_like(query)

    tables = _check_block_tables(block_tables, ctxs, block_size, key_cache)
    t_max = tables.shape[1] * block_size

    # [B, max_blocks, block_size, num_kv_heads, D] -> [B, T_max, kv, D]
    k = key_cache[tables].reshape(batch, t_max, num_kv_heads, head_dim)
    v = value_cache[tables].reshape(batch, t_max, num_kv_heads, head_dim)

    if num_kv_heads != num_heads:
        kv_head_index = torch.arange(num_heads, device=query.device) // heads_per_kv
        k = k[:, :, kv_head_index, :]  # [B, T, H, D]
        v = v[:, :, kv_head_index, :]
    k = k.transpose(1, 2)  # [B, H, T, D]
    v = v.transpose(1, 2)

    q = query.unsqueeze(2)  # [B, H, 1, D]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, H, 1, T]

    # Tokens beyond each sequence's context are masked out (-inf -> prob 0).
    positions = torch.arange(t_max, device=query.device, dtype=ctxs.dtype)
    mask = positions[None, :] >= ctxs[:, None]  # [B, T]
    scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    # Zero the masked V slots so stale pool data cannot leak in as 0 * garbage.
    v = v.masked_fill(mask[:, None, :, None], 0.0)
    out = torch.matmul(probs, v)  # [B, H, 1, D]
    return out.squeeze(2)


def paged_attention_from_cache(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    seq_id: int,
    context_length: int | None = None,
    layer_idx: int = 0,
    scale: float | None = None,
    implementation: str = "torch",
) -> torch.Tensor:
    """Convenience wrapper: paged attention for one sequence in a paged cache.

    Args:
        query: ``[num_heads, D]`` or ``[1, num_heads, D]``.
        kv_cache: the :class:`PagedKVCache` holding this sequence's K/V.
        seq_id: batch row of the sequence inside the cache.
        context_length: number of cached tokens; defaults to the sequence's
            current length.
        layer_idx: which layer's pool to read.
        implementation: backend passed to :func:`paged_attention`.

    Returns:
        ``[1, num_heads, D]`` (or ``[num_heads, D]`` matching the input rank).
    """
    if query.ndim == 2:
        query = query.unsqueeze(0)
    if query.ndim != 3:
        raise ValueError(f"query must be [num_heads, D] or [1, num_heads, D]")
    if seq_id not in kv_cache.sequences:
        raise ValueError(f"sequence {seq_id} is not allocated in the cache")
    if context_length is None:
        context_length = kv_cache.sequences[seq_id].length

    block_tables = kv_cache.get_block_tables()[seq_id:seq_id + 1]
    context_lengths = torch.tensor([context_length], dtype=torch.long, device=query.device)
    out = paged_attention(
        query,
        kv_cache.k_pool[layer_idx],
        kv_cache.v_pool[layer_idx],
        block_tables,
        context_lengths,
        kv_cache.block_size,
        num_kv_heads=kv_cache.num_kv_heads,
        scale=scale,
        implementation=implementation,
    )
    return out
