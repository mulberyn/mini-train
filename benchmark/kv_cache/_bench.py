"""Shared helpers for the KV-cache benchmarks.

The workload models a prefill + decode session at the *cache* level:

* prefill: one update with the whole prompt (``prompt_len`` tokens);
* decode: ``num_decode`` single-token updates (one per step).

The same workload is run against every cache implementation so latency,
throughput, memory and allocation counts are directly comparable.
"""

from __future__ import annotations

import time

import torch


def allocate_rows(cache, batch: int) -> None:
    """Auto-allocate paged-cache sequence slots for rows ``0..batch-1``."""
    if hasattr(cache, "allocate_sequence"):
        for row in range(batch):
            if row not in cache.sequences:
                cache.allocate_sequence(row)


def update_every_layer(cache, key: torch.Tensor, value: torch.Tensor,
                       positions: torch.Tensor) -> None:
    """Update every layer of ``cache`` with the same K/V and positions."""
    for layer_idx in range(cache.num_layers):
        cache.update(layer_idx, key, value, positions)


def run_cache_workload(
    cache,
    *,
    batch: int,
    prompt_len: int,
    num_decode: int,
    warmup: int = 5,
    iterations: int = 20,
) -> dict:
    """Run one prefill + decode session against ``cache``; return metrics."""
    device = cache.device
    head_dim = cache.head_dim
    num_heads = cache.num_kv_heads
    dtype = cache.dtype

    def make_tokens(t_new: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=str(device)).manual_seed(seed)
        return (
            torch.randn(batch, num_heads, t_new, head_dim,
                        dtype=dtype, device=device, generator=generator),
            torch.randn(batch, num_heads, t_new, head_dim,
                        dtype=dtype, device=device, generator=generator),
        )

    def prefill_only() -> None:
        cache.reset()
        allocate_rows(cache, batch)
        key, value = make_tokens(prompt_len, 1)
        update_every_layer(cache, key, value, torch.arange(prompt_len, device=device))

    def full_session() -> None:
        prefill_only()
        for step in range(num_decode):
            key1, value1 = make_tokens(1, step + 100)
            pos = torch.tensor([prompt_len + step], device=device, dtype=torch.long)
            update_every_layer(cache, key1, value1, pos)

    def timed(fn, reps: int) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / reps

    for _ in range(warmup):
        full_session()
    prefill_ms = timed(prefill_only, iterations) * 1000.0
    alloc_before = cache.allocation_count
    session_ms = timed(full_session, iterations) * 1000.0
    alloc_after = cache.allocation_count
    decode_ms_per_token = (session_ms - prefill_ms) / num_decode

    tokens_per_session = prompt_len + num_decode
    allocations_per_session = (alloc_after - alloc_before) / iterations
    return {
        "cache_type": type(cache).__name__.replace("KVCache", ""),
        "batch": batch,
        "prompt_len": prompt_len,
        "num_decode": num_decode,
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": decode_ms_per_token,
        "tokens_per_sec": tokens_per_session / (session_ms / 1000.0),
        "memory_mb": cache.memory_usage() / 1024**2,
        "allocation_count": allocations_per_session,
        "allocations_per_step": allocations_per_session / max(tokens_per_session, 1),
    }


def print_benchmark_header(title: str) -> None:
    print("=" * 78)
    print("miniLLM-engine Benchmark")
    print("=" * 78)
    from utils.utils import print_hardware_info
    print_hardware_info()
    print()
    print(title)
    print("-" * 78)


def print_results(results: list[dict]) -> None:
    print(f"{'Cache':<10}{'B':>3}{'prompt':>8}{'decode':>8}"
          f"{'prefill ms':>12}{'dec ms/tok':>12}{'tok/s':>12}"
          f"{'mem MB':>10}{'alloc':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['cache_type']:<10}{r['batch']:>3}{r['prompt_len']:>8}"
              f"{r['num_decode']:>8}{r['prefill_ms']:>12.3f}"
              f"{r['decode_ms_per_token']:>12.3f}{r['tokens_per_sec']:>12.1f}"
              f"{r['memory_mb']:>10.3f}{r['allocation_count']:>8}")
    print("=" * 78)
