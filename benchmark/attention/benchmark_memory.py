"""Experiment C: KV-cache memory, Dense vs Paged (docs/paged_attention.md 9).

The static cache reserves ``B * max_seq_len`` KV bytes up front for every
sequence; the paged cache allocates only the blocks the sequences actually
use. The paged pool here is sized to the *actual* demand (as a serving engine
would grow its pool), which is what produces the flat memory line.

Run:
    python -m benchmark.attention.benchmark_memory [--device cpu|cuda]
"""

from __future__ import annotations

import argparse

import torch

from inference.kv_cache import PagedKVCache, StaticKVCache
from utils.utils import print_hardware_info

NUM_LAYERS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 16


def cache_memory_mb(cache) -> float:
    return cache.memory_usage() / 1024**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    max_seq_len = 8192 if device.type == "cuda" else 4096
    batch_sizes = [1, 8, 32, 64] if device.type == "cuda" else [1, 8, 32]
    contexts = [512, 1024, 2048, 4096, 8192] if device.type == "cuda" else [512, 1024, 2048, 4096]

    print_hardware_info()
    print()
    print("=" * 78)
    print("KV Cache Memory: Dense (static) vs Paged")
    print("=" * 78)
    print(f"layers={args.num_layers} num_kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} "
          f"max_seq_len={max_seq_len} block_size={BLOCK_SIZE}")
    print()

    common = dict(
        num_layers=args.num_layers,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=dtype,
        device=device,
    )

    print(f"{'B':>4}{'ctx':>7}{'static MB':>12}{'paged MB':>12}{'paged/static':>12}")
    print("-" * 78)
    for batch in batch_sizes:
        for ctx in contexts:
            if ctx > max_seq_len:
                continue
            static = StaticKVCache(
                max_batch_size=batch, max_seq_len=max_seq_len, **common
            )
            # Paged pool sized to the actual demand of this config.
            num_blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE * batch
            paged = PagedKVCache(
                max_batch_size=batch, num_blocks=num_blocks, block_size=BLOCK_SIZE,
                **common,
            )
            for row in range(batch):
                paged.allocate_sequence(row)
            key = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM, device=device, dtype=dtype)
            for t in range(ctx):
                paged.update(0, key, key, torch.tensor([t], device=device))
            static_mb = cache_memory_mb(static)
            paged_mb = cache_memory_mb(paged)
            print(f"{batch:>4}{ctx:>7}{static_mb:>12.2f}{paged_mb:>12.2f}"
                  f"{paged_mb / static_mb:>12.3f}")
    print("=" * 78)
    print("\nStatic reserves B*max_seq_len KV bytes up front regardless of the real")
    print("context; Paged (pool sized to demand) allocates only the blocks used,")
    print("so its memory grows with the real sequence lengths.")


if __name__ == "__main__":
    main()
