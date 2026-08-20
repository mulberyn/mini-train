"""Python Paged Attention (reference implementation).

Given a set of sequences whose K/V live in a paged block pool, compute one
attention output per query token by walking each sequence's *block table*
(logical block -> physical block) and gathering the relevant K/V blocks on
the fly.

    logical block 0 -> physical block 17
    logical block 1 -> physical block 4
    ...

The first version is deliberately written for **correctness**, not speed: it
is a plain Python/PyTorch loop that mirrors what the CUDA kernel will later
do (one query head, iterate KV blocks). The benchmark
``benchmark/attention/benchmark_paged_attention.py`` shows that this paged
path can even be slower than dense attention for a single request -- the win
comes from memory efficiency and concurrency, not from per-request latency.
"""

from __future__ import annotations

import math

import torch

from inference.kv_cache.paged import PagedKVCache


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    block_size: int,
    num_kv_heads: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Paged attention for a batch of single-token queries.

    Args:
        query: ``[B, num_heads, D]`` query tokens (decode step).
        key_cache: ``[num_blocks, block_size, num_kv_heads, D]`` physical pool.
        value_cache: same shape as ``key_cache``.
        block_tables: ``[B, max_num_blocks]`` int64 mapping each sequence's
            logical blocks to physical blocks (``-1`` padding ignored).
        context_lengths: ``[B]`` number of cached tokens per sequence.
        block_size: tokens per block.
        num_kv_heads: KV heads in the pool; defaults to ``query`` heads (MHA).
            For GQA the query heads are grouped (head ``i`` reads KV head
            ``i // (num_heads // num_kv_heads)``).
        scale: score scale; defaults to ``1 / sqrt(D)``.

    Returns:
        ``[B, num_heads, D]`` attention outputs.
    """
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

    outputs = []
    for b in range(batch):
        q = query[b]  # [num_heads, D]
        ctx = int(context_lengths[b])
        if ctx == 0:
            outputs.append(torch.zeros(num_heads, head_dim, dtype=query.dtype, device=query.device))
            continue
        num_logical_blocks = (ctx + block_size - 1) // block_size
        if num_logical_blocks > block_tables.shape[1]:
            raise IndexError(
                f"sequence {b} needs {num_logical_blocks} blocks but block table "
                f"has only {block_tables.shape[1]} columns"
            )

        # Iterate the sequence's logical blocks, mapping through the block
        # table to physical blocks, and trim the last block to ctx.
        k_parts, v_parts = [], []
        for logical in range(num_logical_blocks):
            physical = int(block_tables[b, logical])
            if physical < 0:
                raise RuntimeError(
                    f"sequence {b}: logical block {logical} maps to -1 "
                    f"(block table not fully allocated?)"
                )
            if physical >= key_cache.shape[0]:
                raise IndexError(
                    f"physical block {physical} out of pool range "
                    f"[0, {key_cache.shape[0]})"
                )
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
            # GQA: expand KV heads to query heads by repeating each group.
            kv_head_index = torch.arange(num_heads, device=k.device) // heads_per_kv
            k = k[:, kv_head_index, :]  # [ctx, num_heads, D]
            v = v[:, kv_head_index, :]
        k = k.transpose(0, 1)  # [num_heads, ctx, D]
        v = v.transpose(0, 1)

        # Causal masking is implicit: only the ctx *past + current* tokens are
        # gathered, so every cached key position is eligible.
        scores = torch.einsum("hd,htd->ht", q, k) * scale  # [num_heads, ctx]
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("ht,htd->hd", probs, v)  # [num_heads, D]
        outputs.append(out)

    return torch.stack(outputs, dim=0)


def paged_attention_from_cache(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    seq_id: int,
    context_length: int | None = None,
    layer_idx: int = 0,
    scale: float | None = None,
) -> torch.Tensor:
    """Convenience wrapper: paged attention for one sequence in a paged cache.

    Args:
        query: ``[num_heads, D]`` or ``[1, num_heads, D]``.
        kv_cache: the :class:`PagedKVCache` holding this sequence's K/V.
        seq_id: batch row of the sequence inside the cache.
        context_length: number of cached tokens; defaults to the sequence's
            current length.
        layer_idx: which layer's pool to read.

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
    )
    return out
