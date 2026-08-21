"""Triton paged attention kernel (docs/paged_attention.md Phase 3).

One program per ``(batch, head)`` pair: the batch dimension is parallelized on
the GPU (the fix for the loop implementation's ``O(B)`` Python overhead), and
each program walks its sequence's block table, accumulating attention with a
flash-attention style online softmax:

    for logical_block in range(num_blocks):
        phys = block_table[b, logical_block]
        k_block = K_pool[phys]        # [BLOCK_SIZE, kv_heads, HEAD_DIM]
        ...
        m_new = max(m, scores)
        acc   = acc * exp(m - m_new) + exp(scores - m_new) @ v_block
        l     = l * exp(m - m_new) + sum(exp(scores - m_new))

This is the *naive* Triton version -- correctness first. Obvious next
optimizations (documented but not applied yet): replace the elementwise
scores with ``tl.dot`` (requires BLOCK_SIZE/HEAD_DIM >= 16), split K/V block
iteration across multiple programs per head (sequence parallelism), and use
warp-level reduction for the online softmax.

Requirements
------------
* ``triton`` must be installed (Linux + CUDA). On this Windows host triton is
  unavailable, so importing this module still works and calling the kernel
  raises a clear ``RuntimeError``; the correctness tests auto-skip.
* ``head_dim`` and ``block_size`` must be powers of two (``tl.arange``
  constraint), and ``head_dim`` must divide into the pool layout.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the host platform
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _paged_attention_kernel(
        Q, K, V, BlockTables, ContextLengths, Out,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_num_blocks,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """One program per (batch, head): iterate KV blocks, online softmax."""
        pid = tl.program_id(0)
        b = pid // num_heads
        h = pid % num_heads
        ctx = tl.load(ContextLengths + b).to(tl.int32)

        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(
            Q + b * num_heads * head_dim + h * head_dim + offs_d
        ).to(tl.float32)  # [HEAD_DIM]

        # GQA: query head h reads kv head h // (num_heads // num_kv_heads).
        kv_h = h // (num_heads // num_kv_heads)

        if ctx > 0:
            num_blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
            m_i = tl.full([1], float("-inf"), tl.float32)
            l_i = tl.zeros([1], tl.float32)
            acc = tl.zeros([HEAD_DIM], tl.float32)

            for logical in range(0, num_blocks):
                phys = tl.load(BlockTables + b * max_num_blocks + logical)
                offs_t = tl.arange(0, BLOCK_SIZE)
                # K block slice for this kv head: [BLOCK_SIZE, HEAD_DIM].
                k_ptrs = (
                    K
                    + phys * block_size * num_kv_heads * head_dim
                    + offs_t[:, None] * num_kv_heads * head_dim
                    + kv_h * head_dim
                    + offs_d[None, :]
                )
                k_block = tl.load(k_ptrs).to(tl.float32)
                v_ptrs = (
                    V
                    + phys * block_size * num_kv_heads * head_dim
                    + offs_t[:, None] * num_kv_heads * head_dim
                    + kv_h * head_dim
                    + offs_d[None, :]
                )
                v_block = tl.load(v_ptrs).to(tl.float32)

                # Elementwise dot (naive; tl.dot needs dims >= 16).
                scores = tl.sum(q[None, :] * k_block, axis=1) * scale  # [BLOCK_SIZE]

                # Mask tokens beyond this sequence's context.
                pos = logical * BLOCK_SIZE + offs_t
                scores = tl.where(pos < ctx, scores, float("-inf"))

                m_new = tl.maximum(m_i, tl.max(scores, axis=0))  # scalar [1]
                p = tl.exp(scores - m_new)
                alpha = tl.exp(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
                m_i = m_new

            acc = acc / l_i
        else:
            acc = tl.zeros([HEAD_DIM], tl.float32)

        tl.store(
            Out + b * num_heads * head_dim + h * head_dim + offs_d,
            acc.to(q.dtype),
        )


def paged_attention_triton(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    block_size: int,
    num_kv_heads: int | None = None,
    scale: float | None = None,
    num_warps: int = 4,
) -> torch.Tensor:
    """Triton paged attention (see :func:`paged_attention` for the signature).

    Args:
        num_warps: Triton kernel launch parameter.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "triton is not installed on this host (it needs Linux + CUDA). "
            "The torch ('implementation=\"torch\"') and loop reference "
            "('implementation=\"loop\"') implementations are always available."
        )

    batch, num_heads, head_dim = query.shape
    num_kv_heads = num_kv_heads or num_heads
    if head_dim & (head_dim - 1):
        raise ValueError(f"head_dim must be a power of two, got {head_dim}")
    if block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")

    q = query.contiguous()
    k = key_cache.contiguous()
    v = value_cache.contiguous()
    tables = block_tables.to(q.device).contiguous()
    ctxs = context_lengths.to(q.device).contiguous()
    out = torch.empty_like(q)
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    max_num_blocks = tables.shape[1]

    grid = (batch * num_heads,)
    _paged_attention_kernel[grid](
        q, k, v, tables, ctxs, out,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_num_blocks,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        num_warps=num_warps,
    )
    return out
