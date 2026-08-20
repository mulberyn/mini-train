"""Interface conformance and validation tests shared by every KV cache."""

import pytest
import torch

from tests.inference.conftest import (
    NUM_HEADS,
    NUM_LAYERS,
    any_cache,  # noqa: F401  (pytest fixture)
    update_tokens,
)


def random_chunk(cache, batch, t_new, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, NUM_HEADS, t_new, cache.head_dim, generator=generator)


def test_interface_shapes(any_cache):
    cache = any_cache
    batch = 2
    key = random_chunk(cache, batch, 4, 1)
    value = random_chunk(cache, batch, 4, 2)
    update_tokens(cache, key, value, torch.arange(4))
    for layer_idx in range(NUM_LAYERS):
        k, v = cache.get(layer_idx)
        assert k.shape == (batch, NUM_HEADS, 4, cache.head_dim)
        assert v.shape == (batch, NUM_HEADS, 4, cache.head_dim)
        assert k.dtype == cache.dtype
        assert k.device.type == cache.device.type


def test_update_then_get_returns_same_values(any_cache):
    cache = any_cache
    batch = 2
    if hasattr(cache, "allocate_sequence"):
        for row in range(batch):
            cache.allocate_sequence(row)
    key = random_chunk(cache, batch, 3, 11)
    value = random_chunk(cache, batch, 3, 12)
    update_tokens(cache, key, value, torch.arange(3))
    for layer_idx in range(NUM_LAYERS):
        k, v = cache.get(layer_idx)
        torch.testing.assert_close(k, key)
        torch.testing.assert_close(v, value)


def test_get_positions_slice(any_cache):
    cache = any_cache
    batch = 1
    if hasattr(cache, "allocate_sequence"):
        cache.allocate_sequence(0)
    key = random_chunk(cache, batch, 5, 21)
    value = random_chunk(cache, batch, 5, 22)
    update_tokens(cache, key, value, torch.arange(5))
    for layer_idx in range(NUM_LAYERS):
        k, v = cache.get(layer_idx, torch.tensor([1, 3]))
        # get() pads rows to max_batch_size; only row 0 is populated.
        torch.testing.assert_close(k[:batch, :, 0], key[:, :, 1])
        torch.testing.assert_close(v[:batch, :, 1], value[:, :, 3])
        k1, v1 = cache.get(layer_idx, 2)
        torch.testing.assert_close(k1[:batch, :, 0], key[:, :, 2])


def test_reset_clears_all_layers(any_cache):
    cache = any_cache
    batch = 1
    if hasattr(cache, "allocate_sequence"):
        cache.allocate_sequence(0)
    key = random_chunk(cache, batch, 4, 31)
    value = random_chunk(cache, batch, 4, 32)
    update_tokens(cache, key, value, torch.arange(4))
    cache.reset()
    for layer_idx in range(NUM_LAYERS):
        k, v = cache.get(layer_idx)
        assert k.shape[-2] == 0
        assert v.shape[-2] == 0
    # Caches that keep their buffers (static/dynamic) still report the retained
    # capacity; naive/paged release everything and report zero bytes.
    if type(cache).__name__ in ("NaiveKVCache", "PagedKVCache"):
        assert cache.memory_usage() == 0.0


def test_allocation_count_available(any_cache):
    cache = any_cache
    assert isinstance(cache.allocation_count, int)
    assert cache.allocation_count >= 0


def test_update_rejects_bad_layer(any_cache):
    cache = any_cache
    key = random_chunk(cache, 1, 1, 41)
    with pytest.raises(IndexError):
        cache.update(NUM_LAYERS, key, key, torch.tensor([0]))
    with pytest.raises(IndexError):
        cache.update(-1, key, key, torch.tensor([0]))


def test_update_rejects_shape_mismatch(any_cache):
    cache = any_cache
    key = random_chunk(cache, 1, 1, 51)
    value = random_chunk(cache, 1, 2, 52)
    with pytest.raises(ValueError):
        cache.update(0, key, value, torch.tensor([0]))


def test_update_rejects_bad_heads(any_cache):
    cache = any_cache
    key = torch.randn(1, NUM_HEADS + 1, 1, cache.head_dim)
    with pytest.raises(ValueError):
        cache.update(0, key, key, torch.tensor([0]))


def test_update_rejects_noncontiguous_positions(any_cache):
    cache = any_cache
    key = random_chunk(cache, 1, 2, 61)
    with pytest.raises(ValueError):
        cache.update(0, key, key, torch.tensor([0, 2]))


def test_update_rejects_bad_positions_dtype(any_cache):
    cache = any_cache
    key = random_chunk(cache, 1, 1, 62)
    with pytest.raises(TypeError):
        cache.update(0, key, key, torch.tensor([0.0]))
