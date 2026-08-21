"""Experiment D: heterogeneous sequence lengths (docs/paged_attention.md 10).

The scenario paged attention is really built for: continuous batching with
*unequal* sequence lengths. A dense implementation must pad every sequence to
the longest one (wasted compute); paged attention only touches the blocks a
sequence actually owns.

The dense baseline here pads to the longest context and runs SDPA on the
padded tensor; the paged path uses the real per-sequence block tables.

Run:
    python -m benchmark.attention.benchmark_heterogeneous [--device cpu|cuda]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from benchmark.attention._bench import make_workload, print_attention_header, timeit
from inference.attention import paged_attention

NUM_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 16
NUM_BLOCKS = 16384


def dense_padded_reference(query, key_cache, value_cache, block_tables, context_lengths, block_size):
    """Pad every sequence to the max context, then run SDPA."""
    batch = query.shape[0]
    max_ctx = int(context_lengths.max().item())
    max_blocks = (max_ctx + block_size - 1) // block_size
    k_parts, v_parts = [], []
    for logical in range(max_blocks):
        k_parts.append(key_cache[block_tables[:, logical]])
        v_parts.append(value_cache[block_tables[:, logical]])
    k = torch.cat(k_parts, dim=1)[:, :max_ctx].transpose(1, 2)
    v = torch.cat(v_parts, dim=1)[:, :max_ctx].transpose(1, 2)
    q = query.unsqueeze(2)
    return F.scaled_dot_product_attention(q, k, v).squeeze(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print_attention_header(
        "Heterogeneous batch: Dense+padding vs Paged (real lengths)"
    )

    # Uneven contexts: lots of short sequences + a few long ones.
    base = [128, 256, 512, 1024, 2048, 4096] if device.type == "cuda" else [128, 256, 512, 1024]
    configs = {
        "8 seqs": [128] * 4 + [1024] * 2 + [4096] * 2,
        "16 seqs": [128] * 8 + [1024] * 4 + [4096] * 4,
        "32 seqs": [128] * 16 + [1024] * 8 + [4096] * 8,
    } if device.type == "cuda" else {
        "8 seqs": [128] * 4 + [512] * 2 + [1024] * 2,
        "16 seqs": [128] * 8 + [512] * 4 + [1024] * 4,
    }

    print(f"{'config':>10}{'#seq':>5}{'avg ctx':>8}{'max ctx':>8}"
          f"{'paged ms':>12}{'dense ms':>12}{'speedup':>10}")
    print("-" * 78)
    for name, lengths in configs.items():
        batch = len(lengths)
        ctxs = torch.tensor(lengths, dtype=torch.long)
        query, key_cache, value_cache, block_tables, context_lengths = make_workload(
            batch, ctxs, num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, device=device, dtype=dtype,
        )
        paged_ms = timeit(
            lambda: paged_attention(
                query, key_cache, value_cache, block_tables, context_lengths,
                BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, implementation="torch",
            ),
            device, args.warmup, args.iterations,
        )
        dense_ms = timeit(
            lambda: dense_padded_reference(
                query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE
            ),
            device, args.warmup, args.iterations,
        )
        avg_ctx = int(ctxs.float().mean().item())
        max_ctx = int(ctxs.max().item())
        print(f"{name:>10}{batch:>5}{avg_ctx:>8}{max_ctx:>8}"
              f"{paged_ms:>12.3f}{dense_ms:>12.3f}{dense_ms / paged_ms:>10.2f}x")
    print("=" * 78)
    print("\nThe dense baseline wastes compute padding short sequences up to the")
    print("longest; paged attention touches only the blocks each sequence owns.")


if __name__ == "__main__":
    main()
