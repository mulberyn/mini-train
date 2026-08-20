"""Tests for the naive KV cache (docs/kv_cache.md section 三)."""

import pytest
import torch

from inference.kv_cache import NaiveKVCache
from tests.inference.conftest import (
    CONTEXT_LENGTH,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    make_cache,
    update_tokens,
)


def make_naive(**kwargs):
    return make_cache("naive", **kwargs)


def test_update_appends_sequential_chunks():
    cache = make_naive(max_batch_size=1)
    key1 = torch.randn(1, NUM_HEADS, 3, D_MODEL // NUM_HEADS)
    value1 = torch.randn(1, NUM_HEADS, 3, D_MODEL // NUM_HEADS)
    update_tokens(cache, key1, value1, torch.arange(3))

    key2 = torch.randn(1, NUM_HEADS, 2, D_MODEL // NUM_HEADS)
    value2 = torch.randn(1, NUM_HEADS, 2, D_MODEL // NUM_HEADS)
    update_tokens(cache, key2, value2, torch.arange(3, 5))

    for layer_idx in range(NUM_LAYERS):
        k, v = cache.get(layer_idx)
        assert k.shape == (1, NUM_HEADS, 5, D_MODEL // NUM_HEADS)
        torch.testing.assert_close(k[:, :, :3], key1)
        torch.testing.assert_close(k[:, :, 3:], key2)
        torch.testing.assert_close(v[:, :, 3:], value2)


def test_update_without_positions_appends():
    cache = make_naive(max_batch_size=1)
    key1 = torch.randn(1, NUM_HEADS, 2, D_MODEL // NUM_HEADS)
    update_tokens(cache, key1, key1, None)
    key2 = torch.randn(1, NUM_HEADS, 2, D_MODEL // NUM_HEADS)
    update_tokens(cache, key2, key2, None)
    k, _ = cache.get(0)
    assert k.shape[-2] == 4
    torch.testing.assert_close(k[:, :, :2], key1)
    torch.testing.assert_close(k[:, :, 2:], key2)


def test_update_rejects_gap():
    cache = make_naive(max_batch_size=1)
    key = torch.randn(1, NUM_HEADS, 1, D_MODEL // NUM_HEADS)
    update_tokens(cache, key, key, torch.tensor([0]))
    with pytest.raises(ValueError):
        update_tokens(cache, key, key, torch.tensor([2]))


def test_update_rejects_exceeding_max_seq_len():
    cache = make_naive(max_batch_size=1, max_seq_len=4)
    key = torch.randn(1, NUM_HEADS, 3, D_MODEL // NUM_HEADS)
    update_tokens(cache, key, key, torch.arange(3))
    with pytest.raises(ValueError):
        update_tokens(cache, key, key, torch.arange(3, 6))


def test_memory_usage_grows_with_tokens():
    cache = make_naive(max_batch_size=1)
    per_token_bytes = 2 * NUM_LAYERS * NUM_HEADS * (D_MODEL // NUM_HEADS) * 4
    key = torch.randn(1, NUM_HEADS, 1, D_MODEL // NUM_HEADS)
    assert cache.memory_usage() == 0.0
    for t in range(3):
        update_tokens(cache, key, key, torch.tensor([t]))
        expected = per_token_bytes * (t + 1)
        assert cache.memory_usage() == expected
    cache.reset()
    assert cache.memory_usage() == 0.0


def test_allocation_count_grows_with_updates():
    cache = make_naive(max_batch_size=1)
    assert cache.allocation_count == 0
    key = torch.randn(1, NUM_HEADS, 1, D_MODEL // NUM_HEADS)
    update_tokens(cache, key, key, torch.tensor([0]))
    assert cache.allocation_count == 2 * NUM_LAYERS
    update_tokens(cache, key, key, torch.tensor([1]))
    assert cache.allocation_count == 4 * NUM_LAYERS


def test_multi_batch_multi_layer_consistency():
    cache = make_naive(max_batch_size=2)
    key = torch.randn(2, NUM_HEADS, 2, D_MODEL // NUM_HEADS)
    value = torch.randn(2, NUM_HEADS, 2, D_MODEL // NUM_HEADS)
    update_tokens(cache, key, value, torch.arange(2))
    for layer_idx in range(NUM_LAYERS):
        k, v = cache.get(layer_idx)
        torch.testing.assert_close(k, key)
        torch.testing.assert_close(v, value)
    assert cache.num_layers == NUM_LAYERS


def test_get_before_any_update_returns_empty():
    cache = make_naive(max_batch_size=1)
    k, v = cache.get(0)
    assert k.shape == (1, NUM_HEADS, 0, D_MODEL // NUM_HEADS)
    assert v.shape == (1, NUM_HEADS, 0, D_MODEL // NUM_HEADS)


def test_constructor_validation():
    with pytest.raises(ValueError):
        NaiveKVCache(0, 1, 16, NUM_HEADS, 8, torch.float32, "cpu")
    with pytest.raises(ValueError):
        NaiveKVCache(NUM_LAYERS, 0, 16, NUM_HEADS, 8, torch.float32, "cpu")
    with pytest.raises(ValueError):
        NaiveKVCache(NUM_LAYERS, 1, 0, NUM_HEADS, 8, torch.float32, "cpu")
