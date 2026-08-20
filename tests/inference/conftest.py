"""Shared fixtures/helpers for the inference KV-cache test suite."""

from __future__ import annotations

import pytest
import torch

from inference.kv_cache import (
    DynamicKVCache,
    KVCache,
    NaiveKVCache,
    PagedKVCache,
    StaticKVCache,
)
from inference.model_runner import ModelRunner
from trainer.model.transformer import TransformerLM

VOCAB_SIZE = 128
CONTEXT_LENGTH = 32
D_MODEL = 32
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 64
ROPE_THETA = 10000.0

MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "context_length": CONTEXT_LENGTH,
    "d_model": D_MODEL,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "d_ff": D_FF,
    "rope_theta": ROPE_THETA,
}


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % VOCAB_SIZE for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(int(i)) for i in ids)


def make_model(device="cpu", **overrides) -> TransformerLM:
    config = {**MODEL_CONFIG, **overrides}
    return TransformerLM(
        **config, device=device, dtype=torch.float32
    )


def make_runner(device="cpu", **model_overrides) -> ModelRunner:
    model = make_model(device, **model_overrides)
    return ModelRunner(model=model, tokenizer=DummyTokenizer(), device=device)


def make_cache(
    cache_type: str,
    *,
    max_batch_size: int = 1,
    max_seq_len: int = CONTEXT_LENGTH,
    num_blocks: int = 32,
    block_size: int = 8,
    device="cpu",
    **kwargs,
) -> KVCache:
    """Build a KV cache with the standard test-model geometry."""
    common = dict(
        num_layers=NUM_LAYERS,
        max_batch_size=max_batch_size,
        num_kv_heads=NUM_HEADS,
        head_dim=D_MODEL // NUM_HEADS,
        dtype=torch.float32,
        device=torch.device(device),
    )
    if cache_type == "naive":
        return NaiveKVCache(max_seq_len=max_seq_len, **common, **kwargs)
    if cache_type == "static":
        return StaticKVCache(max_seq_len=max_seq_len, **common, **kwargs)
    if cache_type == "dynamic":
        return DynamicKVCache(max_seq_len=max_seq_len, **common, **kwargs)
    if cache_type == "paged":
        return PagedKVCache(
            num_blocks=num_blocks, block_size=block_size, **common, **kwargs
        )
    raise ValueError(f"unknown cache_type {cache_type!r}")


ALL_CACHE_TYPES = ["naive", "static", "dynamic", "paged"]


@pytest.fixture(params=ALL_CACHE_TYPES, ids=ALL_CACHE_TYPES)
def any_cache(request) -> KVCache:
    """Parametrized fixture: every concrete KVCache implementation."""
    return make_cache(request.param, max_batch_size=2, num_blocks=64)


def update_tokens(
    cache: KVCache,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> None:
    """Update every layer of a cache with the same K/V and positions.

    Paged caches get their batch rows auto-allocated first (mirroring
    ``ModelRunner.prefill``); the dedicated paged tests exercise allocation
    explicitly.
    """
    if hasattr(cache, "allocate_sequence"):
        for row in range(key.size(0)):
            if row not in cache.sequences:
                cache.allocate_sequence(row)
    for layer_idx in range(cache.num_layers):
        cache.update(layer_idx, key, value, positions)
