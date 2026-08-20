"""Tests for the static KV cache (docs/kv_cache.md section 五)."""

import torch

from inference.kv_cache import StaticKVCache
from tests.inference.conftest import (
    CONTEXT_LENGTH,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    make_cache,
    update_tokens,
)

HEAD_DIM = D_MODEL // NUM_HEADS


def make_static(**kwargs):
    return make_cache("static", **kwargs)


def test_update_writes_at_positions():
    cache = make_static(max_batch_size=1, max_seq_len=16)
    key1 = torch.randn(1, NUM_HEADS, 3, HEAD_DIM)
    value1 = torch.randn(1, NUM_HEADS, 3, HEAD_DIM)
    update_tokens(cache, key1, value1, torch.arange(3))

    key2 = torch.randn(1, NUM_HEADS, 2, HEAD_DIM)
    value2 = torch.randn(1, NUM_HEADS, 2, HEAD_DIM)
    update_tokens(cache, key2, value2, torch.arange(3, 5))

    for layer_idx in range(NUM_LAYERS):
        k, v = cache.get(layer_idx)
        assert k.shape == (1, NUM_HEADS, 5, HEAD_DIM)
        torch.testing.assert_close(k[:, :, :3], key1)
        torch.testing.assert_close(k[:, :, 3:], key2)
        torch.testing.assert_close(v[:, :, 3:], value2)


def test_update_without_positions_appends_per_row():
    cache = make_static(max_batch_size=2, max_seq_len=16)
    key = torch.randn(2, NUM_HEADS, 2, HEAD_DIM)
    update_tokens(cache, key, key, None)
    # Row 0 advances one extra token (single-row, single-token update).
    one = key[0:1, :, 0:1, :]
    cache.update(0, one, one, None)
    k, _ = cache.get(0)
    assert k.shape == (2, NUM_HEADS, 3, HEAD_DIM)
    torch.testing.assert_close(k[0, :, 2:3], one[0])
    assert torch.equal(k[1, :, 2:3], torch.zeros_like(k[1, :, 2:3]))


def test_get_returns_valid_prefix_only():
    cache = make_static(max_batch_size=1, max_seq_len=16)
    key = torch.randn(1, NUM_HEADS, 4, HEAD_DIM)
    update_tokens(cache, key, key, torch.arange(4))
    k, _ = cache.get(0)
    assert k.shape[-2] == 4
    assert cache.k_cache[0].shape[-2] == 16  # buffer stays fully allocated


def test_update_rejects_exceeding_max_seq_len():
    import pytest
    cache = make_static(max_batch_size=1, max_seq_len=4)
    key = torch.randn(1, NUM_HEADS, 3, HEAD_DIM)
    with pytest.raises(ValueError):
        update_tokens(cache, key, key, torch.arange(3, 6))


def test_memory_usage_is_full_capacity():
    cache = make_static(max_batch_size=1, max_seq_len=16)
    per_tensor_bytes = 1 * NUM_HEADS * 16 * HEAD_DIM * 4
    assert cache.memory_usage() == 2 * NUM_LAYERS * per_tensor_bytes
    # Even an empty cache already owns the whole buffer.
    assert cache.memory_usage() > 0.0


def test_allocation_count_is_constant():
    cache = make_static(max_batch_size=1, max_seq_len=16)
    assert cache.allocation_count == 2 * NUM_LAYERS
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    for t in range(8):
        update_tokens(cache, key, key, torch.tensor([t]))
    assert cache.allocation_count == 2 * NUM_LAYERS  # no per-step allocations


def test_reset_zeros_buffer():
    cache = make_static(max_batch_size=1, max_seq_len=16)
    key = torch.randn(1, NUM_HEADS, 4, HEAD_DIM)
    update_tokens(cache, key, key, torch.arange(4))
    cache.reset()
    k, _ = cache.get(0)
    assert k.shape[-2] == 0
    # The buffer itself is zeroed.
    torch.testing.assert_close(cache.k_cache[0], torch.zeros_like(cache.k_cache[0]))


def test_constructor_validation():
    import pytest
    with pytest.raises(ValueError):
        StaticKVCache(NUM_LAYERS, 1, 0, NUM_HEADS, HEAD_DIM, torch.float32, "cpu")
    with pytest.raises(ValueError):
        StaticKVCache(0, 1, 16, NUM_HEADS, HEAD_DIM, torch.float32, "cpu")
