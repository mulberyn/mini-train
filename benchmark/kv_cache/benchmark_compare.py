"""Benchmark: Naive vs Static vs Dynamic vs Paged KV cache.

Runs the identical prefill + decode workload against every KV cache
implementation and prints latency / throughput / memory / allocation counts
(docs/kv_cache.md sections 五 and 二十一).

Run:
    python -m benchmark.kv_cache.benchmark_compare [--device cpu|cuda]
"""

from __future__ import annotations

import argparse

import torch

from benchmark.kv_cache._bench import (
    print_benchmark_header,
    print_results,
    run_cache_workload,
)
from inference.kv_cache import DynamicKVCache, NaiveKVCache, PagedKVCache, StaticKVCache


def make_caches(device, num_layers, batch, max_seq_len, block_size, dtype):
    common = dict(
        num_layers=num_layers,
        max_batch_size=batch,
        num_kv_heads=8,
        head_dim=64,
        dtype=dtype,
        device=device,
    )
    return [
        NaiveKVCache(max_seq_len=max_seq_len, **common),
        StaticKVCache(max_seq_len=max_seq_len, **common),
        DynamicKVCache(max_seq_len=max_seq_len, initial_capacity=64, **common),
        PagedKVCache(
            num_blocks=(max_seq_len + block_size - 1) // block_size * batch,
            block_size=block_size,
            **common,
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-decode", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print_benchmark_header("KV Cache: Naive vs Static vs Dynamic vs Paged")

    caches = make_caches(
        device=device,
        num_layers=args.num_layers,
        batch=args.batch,
        max_seq_len=args.prompt_len + args.num_decode,
        block_size=args.block_size,
        dtype=dtype,
    )
    results = [
        run_cache_workload(
            cache,
            batch=args.batch,
            prompt_len=args.prompt_len,
            num_decode=args.num_decode,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for cache in caches
    ]
    print_results(results)

    print("\nNotes")
    print("-" * 78)
    print("* allocation_count: K/V tensor allocations (excluding the initial pool).")
    print("* Naive reallocates every step (O(T)); Static/Paged allocate once up front;")
    print("  Dynamic allocates only when capacity doubles (O(log T)).")
    print("* memory_mb: bytes occupied after the session (Static = full capacity).")


if __name__ == "__main__":
    main()
