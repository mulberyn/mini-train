"""Profile the KV cache implementations with torch.profiler.

Shows where the time goes for naive (torch.cat per step) vs static (in-place
writes) vs paged (per-token scatter writes), and reports CUDA memory when
running on GPU.

Run:
    python -m profiler.kv_cache.profile_kv_cache [--device cpu|cuda]
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from inference.kv_cache import DynamicKVCache, NaiveKVCache, PagedKVCache, StaticKVCache
from utils.utils import print_hardware_info


def build_session(cache, batch, prompt_len, num_decode):
    """A prefill + decode session on the given cache."""

    def session():
        if hasattr(cache, "allocate_sequence"):
            for row in range(batch):
                if row not in cache.sequences:
                    cache.allocate_sequence(row)
        for layer_idx in range(cache.num_layers):
            key = torch.randn(batch, cache.num_kv_heads, prompt_len, cache.head_dim,
                              dtype=cache.dtype, device=cache.device)
            cache.update(layer_idx, key, key, torch.arange(prompt_len, device=cache.device))
            for step in range(num_decode):
                key1 = torch.randn(batch, cache.num_kv_heads, 1, cache.head_dim,
                                   dtype=cache.dtype, device=cache.device)
                pos = torch.tensor([prompt_len + step], device=cache.device, dtype=torch.long)
                cache.update(layer_idx, key1, key1, pos)

    return session


def profile_cache(name, cache, batch, prompt_len, num_decode, device):
    session = build_session(cache, batch, prompt_len, num_decode)
    for _ in range(3):
        cache.reset()
        session()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, profile_memory=True) as prof:
        with record_function(f"{name}_prefill_decode"):
            cache.reset()
            session()

    print()
    print("=" * 100)
    print(f"{name}")
    print("=" * 100)
    print(
        prof.key_averages().table(
            sort_by=("cuda_time_total" if device.type == "cuda" else "cpu_time_total"),
            row_limit=15,
        )
    )
    if device.type == "cuda":
        print(f"CUDA memory allocated : {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA memory reserved  : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--num-decode", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print_hardware_info()

    max_seq_len = args.prompt_len + args.num_decode
    common = dict(
        num_layers=args.num_layers,
        max_batch_size=args.batch,
        num_kv_heads=8,
        head_dim=64,
        dtype=dtype,
        device=device,
    )
    caches = {
        "NaiveKVCache": NaiveKVCache(max_seq_len=max_seq_len, **common),
        "StaticKVCache": StaticKVCache(max_seq_len=max_seq_len, **common),
        "DynamicKVCache": DynamicKVCache(max_seq_len=max_seq_len, **common),
        "PagedKVCache": PagedKVCache(
            num_blocks=(max_seq_len + args.block_size - 1) // args.block_size * args.batch,
            block_size=args.block_size,
            **common,
        ),
    }
    for name, cache in caches.items():
        profile_cache(name, cache, args.batch, args.prompt_len, args.num_decode, device)


if __name__ == "__main__":
    main()
