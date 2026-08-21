"""Benchmark: Paged Attention implementations vs Dense SDPA.

Compares the loop reference, the batch-vectorized torch implementation and
(where triton is available) the Triton kernel against dense PyTorch SDPA
across batch sizes and context lengths (docs/paged_attention.md 11).

Run:
    python -m benchmark.attention.benchmark_paged_attention [--device cpu|cuda]
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

NUM_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 16
NUM_BLOCKS = 8192


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print_attention_header("Paged Attention (loop / torch / triton) vs Dense SDPA")

    impls = [("loop", "loop"), ("torch", "torch")]
    if TRITON_AVAILABLE:
        impls.append(("triton", "triton"))

    batch_sizes = [1, 8, 32, 64] if device.type == "cuda" else [1, 8, 16, 32]
    seq_lens = [128, 512, 1024, 2048, 4096] if device.type == "cuda" else [128, 512, 1024]

    print(f"{'B':>4}{'ctx':>7}" + "".join(f"{name:>12}" for _, name in impls)
          + f"{'dense':>12}{'torch/dense':>13}")
    print("-" * 78)
    for batch in batch_sizes:
        for ctx in seq_lens:
            ctxs = torch.full((batch,), ctx, dtype=torch.long)
            query, key_cache, value_cache, block_tables, context_lengths = make_workload(
                batch, ctxs, num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
                num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, device=device, dtype=dtype,
            )
            row = f"{batch:>4}{ctx:>7}"
            latencies = {}
            for impl, name in impls:
                ms = timeit(
                    lambda: paged_attention(
                        query, key_cache, value_cache, block_tables, context_lengths,
                        BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, implementation=impl,
                    ),
                    device, args.warmup, args.iterations,
                )
                latencies[impl] = ms
                row += f"{ms:>12.3f}"
            dense_ms = timeit(
                lambda: dense_reference_fn(
                    query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE
                ),
                device, args.warmup, args.iterations,
            )
            ratio = dense_ms / latencies["torch"]
            row += f"{dense_ms:>12.3f}{ratio:>13.2f}x"
            print(row)
    print("=" * 78)
    print("\nExpected: 'loop' degrades ~linearly with B (kernel launches per")
    print("sequence); 'torch' and 'triton' stay GPU-parallel over the batch.")


if __name__ == "__main__":
    main()
