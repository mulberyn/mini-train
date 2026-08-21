"""Shared helpers for the paged-attention benchmarks (docs/paged_attention.md 7-10)."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def make_workload(
    batch: int,
    context_lengths: torch.Tensor,
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
):
    """Build random pools / queries / block tables for a benchmark config."""
    torch.manual_seed(seed)
    max_blocks = max(
        (int(ctx) + block_size - 1) // block_size for ctx in context_lengths.tolist()
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.randint(
        0, num_blocks, (batch, max_blocks), device=device, dtype=torch.long
    )
    query = torch.randn(batch, num_kv_heads, head_dim, device=device, dtype=dtype)
    return query, key_cache, value_cache, block_tables, context_lengths.to(device)


def timeit(fn, device, warmup: int = 5, iterations: int = 30) -> float:
    """Return average latency in milliseconds."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations * 1000.0


def dense_reference_fn(
    query, key_cache, value_cache, block_tables, context_lengths, block_size
):
    """Dense baseline: gather K/V through the block table, run PyTorch SDPA."""
    batch = query.shape[0]
    max_blocks = block_tables.shape[1]
    k_parts, v_parts = [], []
    for logical in range(max_blocks):
        k_parts.append(key_cache[block_tables[:, logical]])
        v_parts.append(value_cache[block_tables[:, logical]])
    k = torch.cat(k_parts, dim=1)  # [B, max_blocks*bs, kv, D]
    v = torch.cat(v_parts, dim=1)
    max_ctx = int(context_lengths.max().item())
    k = k[:, :max_ctx].transpose(1, 2)  # [B, kv, T, D]
    v = v[:, :max_ctx].transpose(1, 2)
    q = query.unsqueeze(2)  # [B, H, 1, D]
    return F.scaled_dot_product_attention(q, k, v).squeeze(2)


def print_attention_header(title: str) -> None:
    print("=" * 78)
    print("miniLLM-engine Benchmark")
    print("=" * 78)
    from utils.utils import print_hardware_info
    print_hardware_info()
    print()
    print(title)
    print("-" * 78)
