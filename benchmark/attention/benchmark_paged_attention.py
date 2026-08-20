"""Benchmark: Python Paged Attention vs Dense SDPA (docs/kv_cache.md 十四).

The experiment from the docs: paged attention is *not* faster than dense
attention for a single request -- its advantage is memory efficiency and
concurrency. This benchmark sweeps batch size and context length and prints
the latency of both paths so the crossover can be observed.

Run:
    python -m benchmark.attention.benchmark_paged_attention [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from inference.attention.paged_attention import paged_attention
from utils.utils import print_hardware_info

NUM_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 16
NUM_BLOCKS = 8192


def make_workload(batch, context_length, device, dtype, seed=0):
    torch.manual_seed(seed)
    num_blocks_needed = (context_length + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (batch, num_blocks_needed), device=device)
    context_lengths = torch.full((batch,), context_length, dtype=torch.long, device=device)
    query = torch.randn(batch, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    value_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    return query, key_cache, value_cache, block_tables, context_lengths


def dense_reference(query, key_cache, value_cache, block_tables, context_lengths):
    """Gather dense K/V per sequence then run PyTorch SDPA."""
    batch = query.shape[0]
    max_blocks = block_tables.shape[1]
    k_parts, v_parts = [], []
    for logical in range(max_blocks):
        k_parts.append(key_cache[block_tables[:, logical]])   # [B, bs, kv, D]
        v_parts.append(value_cache[block_tables[:, logical]])
    k = torch.cat(k_parts, dim=1)   # [B, max_blocks*bs, kv, D]
    v = torch.cat(v_parts, dim=1)
    ctx = int(context_lengths[0])
    k = k[:, :ctx].transpose(1, 2)  # [B, kv, T, D]
    v = v[:, :ctx].transpose(1, 2)
    q = query.unsqueeze(2)          # [B, H, 1, D]
    return F.scaled_dot_product_attention(q, k, v).squeeze(2)


def timeit(fn, device, warmup=5, iterations=30):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16

    print("=" * 78)
    print("miniLLM-engine Benchmark: Paged Attention vs Dense SDPA")
    print("=" * 78)
    print_hardware_info()

    batch_sizes = [1, 8, 32, 64] if device.type == "cuda" else [1, 8, 16, 32]
    seq_lens = [128, 512, 1024, 2048, 4096] if device.type == "cuda" else [128, 512, 1024]

    print()
    print(f"{'B':>4}{'ctx':>7}{'paged ms':>12}{'dense ms':>12}"
          f"{'paged tok/s':>14}{'dense tok/s':>14}{'speedup':>10}")
    print("-" * 78)
    for batch in batch_sizes:
        for ctx in seq_lens:
            query, key_cache, value_cache, block_tables, context_lengths = make_workload(
                batch, ctx, device, dtype
            )
            paged_ms = timeit(
                lambda: paged_attention(
                    query, key_cache, value_cache, block_tables, context_lengths,
                    BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
                ),
                device, args.warmup, args.iterations,
            )
            dense_ms = timeit(
                lambda: dense_reference(
                    query, key_cache, value_cache, block_tables, context_lengths
                ),
                device, args.warmup, args.iterations,
            )
            tokens = batch * ctx
            print(f"{batch:>4}{ctx:>7}{paged_ms:>12.3f}{dense_ms:>12.3f}"
                  f"{tokens / (paged_ms / 1000):>14.0f}{tokens / (dense_ms / 1000):>14.0f}"
                  f"{dense_ms / paged_ms:>10.2f}x")
    print("=" * 78)
    print("\nNote: the Python paged attention is a correctness-first reference;")
    print("for a single request dense SDPA is typically faster. The paged path wins")
    print("through memory efficiency at high concurrency, which a CUDA kernel makes")
    print("fast as well (next phase).")


if __name__ == "__main__":
    main()
