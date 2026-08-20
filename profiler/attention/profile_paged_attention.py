"""Profile Python Paged Attention vs dense SDPA with torch.profiler.

Run:
    python -m profiler.attention.profile_paged_attention [--device cpu|cuda]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from inference.attention.paged_attention import paged_attention
from utils.utils import print_hardware_info

NUM_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 16
NUM_BLOCKS = 4096


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print_hardware_info()

    batch, ctx = args.batch, args.context_length
    num_blocks_needed = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    torch.manual_seed(0)
    query = torch.randn(batch, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    value_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    block_tables = torch.randint(0, NUM_BLOCKS, (batch, num_blocks_needed), device=device)
    context_lengths = torch.full((batch,), ctx, dtype=torch.long, device=device)

    def paged():
        return paged_attention(
            query, key_cache, value_cache, block_tables, context_lengths,
            BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS,
        )

    def dense():
        k_parts, v_parts = [], []
        for logical in range(num_blocks_needed):
            k_parts.append(key_cache[block_tables[:, logical]])
            v_parts.append(value_cache[block_tables[:, logical]])
        k = torch.cat(k_parts, dim=1)[:, :ctx].transpose(1, 2)
        v = torch.cat(v_parts, dim=1)[:, :ctx].transpose(1, 2)
        return F.scaled_dot_product_attention(query.unsqueeze(2), k, v)

    for _ in range(3):
        paged()
        dense()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function("paged_attention"):
            paged()
        with record_function("dense_sdpa"):
            dense()

    print()
    print("=" * 100)
    print("Profiler Summary")
    print("=" * 100)
    print(
        prof.key_averages().table(
            sort_by=("cuda_time_total" if device.type == "cuda" else "cpu_time_total"),
            row_limit=30,
        )
    )


if __name__ == "__main__":
    main()
