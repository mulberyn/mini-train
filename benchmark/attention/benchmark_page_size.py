"""Experiment B: page-size scaling (docs/paged_attention.md section 8).

page_size too small  -> more blocks, more block-table indirection
page_size too large  -> more KV memory fragmentation / wasted slots

There is typically a sweet spot. This benchmark sweeps page sizes at a fixed
context and reports latency (torch paged + dense) plus the KV memory used.

Run:
    python -m benchmark.attention.benchmark_page_size [--device cpu|cuda]
"""

from __future__ import annotations

import argparse

import torch

from benchmark.attention._bench import (
    dense_reference_fn,
    make_workload,
    print_attention_header,
    timeit,
)
from inference.attention import paged_attention

NUM_KV_HEADS = 8
HEAD_DIM = 64
NUM_BLOCKS = 16384
BATCH = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--context", type=int, default=3000)  # not page-aligned
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print_attention_header(
        f"Page-size scaling (context = {args.context}, B = {args.batch})"
    )

    page_sizes = [8, 16, 32, 64, 128, 256] if device.type == "cuda" else [8, 16, 32, 64]

    dtype_size = torch.finfo(dtype).bits // 8
    print(f"{'page':>6}{'#blocks':>9}{'torch ms':>12}{'dense ms':>12}{'waste MB':>10}")
    print("-" * 78)
    for page_size in page_sizes:
        ctxs = torch.full((args.batch,), args.context, dtype=torch.long)
        query, key_cache, value_cache, block_tables, context_lengths = make_workload(
            args.batch, ctxs, num_blocks=NUM_BLOCKS, block_size=page_size,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, device=device, dtype=dtype,
        )
        torch_ms = timeit(
            lambda: paged_attention(
                query, key_cache, value_cache, block_tables, context_lengths,
                page_size, num_kv_heads=NUM_KV_HEADS, implementation="torch",
            ),
            device, args.warmup, args.iterations,
        )
        dense_ms = timeit(
            lambda: dense_reference_fn(
                query, key_cache, value_cache, block_tables, context_lengths, page_size
            ),
            device, args.warmup, args.iterations,
        )
        # Fragmentation: tokens wasted in the last partial block of every
        # sequence, converted to KV bytes.
        n_blocks = (args.context + page_size - 1) // page_size
        waste_tokens = (n_blocks * page_size - args.context) * args.batch
        waste_mb = (
            waste_tokens * 2 * NUM_KV_HEADS * HEAD_DIM * dtype_size / 1024**2
        )
        print(f"{page_size:>6}{n_blocks:>9}{torch_ms:>12.3f}{dense_ms:>12.3f}"
              f"{waste_mb:>10.4f}")
    print("=" * 78)
    print("\nNote: page too small -> more blocks / indirection / random access;")
    print("page too large -> more KV fragmentation (waste MB). A sweet spot")
    print("balances the two.")


if __name__ == "__main__":
    main()
