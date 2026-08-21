"""Profile paged attention implementations vs dense SDPA with torch.profiler.

Shows where the time goes for the loop reference (per-sequence kernels),
the batch-vectorized torch implementation (gather + batched matmuls) and,
where triton is available, the native Triton kernel -- plus kernel launch
counts, which is exactly the metric that separates the loop from the others.

Run:
    python -m profiler.attention.profile_paged_attention [--device cpu|cuda]
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from benchmark.attention._bench import dense_reference_fn, make_workload
from inference.attention import TRITON_AVAILABLE, paged_attention
from utils.utils import print_hardware_info

NUM_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 16
NUM_BLOCKS = 4096


def profile_one(name, fn, device, activities):
    with profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof:
        with record_function(name):
            for _ in range(10):
                fn()
    print()
    print("=" * 100)
    print(f"{name}")
    print("=" * 100)
    print(
        prof.key_averages().table(
            sort_by=("cuda_time_total" if device.type == "cuda" else "cpu_time_total"),
            row_limit=18,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print_hardware_info()

    ctxs = torch.full((args.batch,), args.context_length, dtype=torch.long)
    query, key_cache, value_cache, block_tables, context_lengths = make_workload(
        args.batch, ctxs, num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, device=device, dtype=dtype,
    )

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    fns = [
        ("paged_attention_loop", lambda: paged_attention(
            query, key_cache, value_cache, block_tables, context_lengths,
            BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, implementation="loop")),
        ("paged_attention_torch", lambda: paged_attention(
            query, key_cache, value_cache, block_tables, context_lengths,
            BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, implementation="torch")),
        ("dense_sdpa", lambda: dense_reference_fn(
            query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)),
    ]
    if TRITON_AVAILABLE:
        fns.insert(
            2,
            ("paged_attention_triton", lambda: paged_attention(
                query, key_cache, value_cache, block_tables, context_lengths,
                BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, implementation="triton")),
        )

    for name, fn in fns:
        for _ in range(3):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        profile_one(name, fn, device, activities)


if __name__ == "__main__":
    main()
