"""Experiment A: batch-size scaling (docs/paged_attention.md section 7).

The core experiment of the whole doc: the *loop* paged attention scales as
O(B) (one Python loop + kernels per sequence), while the *torch* (vectorized)
implementation and PyTorch SDPA stay ~flat as the batch grows.

Run:
    python -m benchmark.attention.benchmark_batch_scaling [--device cpu|cuda]
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
from inference.attention import TRITON_AVAILABLE, paged_attention

NUM_KV_HEADS = 16
HEAD_DIM = 64
BLOCK_SIZE = 16
NUM_BLOCKS = 8192
CONTEXT = 2048


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--context", type=int, default=CONTEXT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print_attention_header(
        f"Batch scaling (context = {args.context}, heads = {NUM_KV_HEADS}, "
        f"head_dim = {HEAD_DIM})"
    )

    batch_sizes = [1, 2, 4, 8, 16, 32, 64] if device.type == "cuda" else [1, 2, 4, 8]

    impls = [("loop", "loop"), ("torch", "torch")]
    if TRITON_AVAILABLE:
        impls.append(("triton", "triton"))

    print(f"{'B':>4}{'ctx':>7}" + "".join(f"{name:>14}" for _, name in impls) + f"{'dense':>14}")
    print("-" * 78)
    for batch in batch_sizes:
        ctxs = torch.full((batch,), args.context, dtype=torch.long)
        query, key_cache, value_cache, block_tables, context_lengths = make_workload(
            batch, ctxs, num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, device=device, dtype=dtype,
        )
        row = f"{batch:>4}{args.context:>7}"
        for impl, _ in impls:
            ms = timeit(
                lambda: paged_attention(
                    query, key_cache, value_cache, block_tables, context_lengths,
                    BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, implementation=impl,
                ),
                device, args.warmup, args.iterations,
            )
            row += f"{ms:>14.3f}"
        dense_ms = timeit(
            lambda: dense_reference_fn(
                query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE
            ),
            device, args.warmup, args.iterations,
        )
        row += f"{dense_ms:>14.3f}"
        print(row)
    print("=" * 78)
    print("\nExpected: 'loop' grows ~linearly with B (one kernel launch per")
    print("sequence); 'torch' and 'dense' stay ~flat (GPU-parallel over batch).")


if __name__ == "__main__":
    main()
