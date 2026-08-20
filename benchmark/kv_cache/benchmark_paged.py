"""Benchmark the paged KV cache alone (docs/kv_cache.md 八/九).

Run:
    python -m benchmark.kv_cache.benchmark_paged [--device cpu|cuda]
"""

from __future__ import annotations

import argparse

import torch

from benchmark.kv_cache._bench import (
    print_benchmark_header,
    print_results,
    run_cache_workload,
)
from inference.kv_cache import PagedKVCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-decode", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print_benchmark_header("Paged KV Cache")

    max_seq_len = args.prompt_len + args.num_decode
    cache = PagedKVCache(
        num_layers=args.num_layers,
        max_batch_size=args.batch,
        num_blocks=(max_seq_len + args.block_size - 1) // args.block_size * args.batch,
        block_size=args.block_size,
        num_kv_heads=8,
        head_dim=64,
        dtype=dtype,
        device=device,
    )
    result = run_cache_workload(
        cache, batch=args.batch, prompt_len=args.prompt_len,
        num_decode=args.num_decode, warmup=args.warmup, iterations=args.iterations,
    )
    print_results([result])


if __name__ == "__main__":
    main()
