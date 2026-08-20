"""Tests for the dynamic KV cache (grow-by-doubling capacity)."""

import torch

from inference.kv_cache import DynamicKVCache
from tests.inference.conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    make_cache,
    update_tokens,
)

HEAD_DIM = D_MODEL // NUM_HEADS


def make_dynamic(**kwargs):
    return make_cache("dynamic", **kwargs)


def test_grows_capacity_on_demand():
    cache = make_dynamic(max_batch_size=1, max_seq_len=256, initial_capacity=8)
    assert cache.capacities[0] == 8
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    for t in range(20):
        update_tokens(cache, key, key, torch.tensor([t]))
    # Capacity must have doubled several times (8 -> 16 -> 32) and never be
    # below the used length.
    assert cache.capacities[0] >= 20
    assert cache.capacities[0] <= 32
    k, _ = cache.get(0)
    assert k.shape[-2] == 20


def test_growth_is_bounded_by_max_seq_len():
    import pytest
    cache = make_dynamic(max_batch_size=1, max_seq_len=16, initial_capacity=4)
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    for t in range(16):
        update_tokens(cache, key, key, torch.tensor([t]))
    assert cache.capacities[0] == 16
    with pytest.raises(ValueError):
        update_tokens(cache, key, key, torch.tensor([16]))


def test_content_matches_naive():
    """Dynamic and naive caches must store identical K/V content."""
    from inference.kv_cache import NaiveKVCache

    dynamic = make_dynamic(max_batch_size=1, max_seq_len=64, initial_capacity=4)
    naive = NaiveKVCache(
        num_layers=NUM_LAYERS, max_batch_size=1, max_seq_len=64,
        num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM, dtype=torch.float32, device="cpu",
    )
    torch.manual_seed(0)
    for t in range(9):
        key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        value = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        update_tokens(dynamic, key, value, torch.tensor([t]))
        update_tokens(naive, key, value, torch.tensor([t]))
    for layer_idx in range(NUM_LAYERS):
        kd, vd = dynamic.get(layer_idx)
        kn, vn = naive.get(layer_idx)
        torch.testing.assert_close(kd, kn)
        torch.testing.assert_close(vd, vn)


def test_allocation_count_between_naive_and_static():
    """Dynamic must allocate far fewer tensors than naive over many steps."""
    dynamic = make_dynamic(max_batch_size=1, max_seq_len=1024, initial_capacity=16)
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    steps = 200
    for t in range(steps):
        update_tokens(dynamic, key, key, torch.tensor([t]))
    # naive would need 2*num_layers*steps allocations; dynamic only allocates
    # on growth (log2(1024/16) = 6 doublings -> 12 tensors).
    assert dynamic.allocation_count < 2 * NUM_LAYERS * steps
    assert dynamic.allocation_count <= 2 * NUM_LAYERS * 7


def test_reset_keeps_capacity():
    cache = make_dynamic(max_batch_size=1, max_seq_len=64, initial_capacity=4)
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    for t in range(20):
        update_tokens(cache, key, key, torch.tensor([t]))
    grown = cache.capacities[0]
    cache.reset()
    k, _ = cache.get(0)
    assert k.shape[-2] == 0
    assert cache.capacities[0] == grown  # capacity is retained after reset


def test_memory_usage_reflects_current_capacity():
    cache = make_dynamic(max_batch_size=1, max_seq_len=64, initial_capacity=4)
    initial = cache.memory_usage()
    per_layer_bytes = 2 * 1 * NUM_HEADS * 4 * HEAD_DIM * 4
    assert initial == NUM_LAYERS * per_layer_bytes
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    for t in range(10):
        update_tokens(cache, key, key, torch.tensor([t]))
    # 4 -> 8 -> 16 (10 tokens need capacity >= 10).
    assert cache.capacities[0] == 16
    grown = 2 * 1 * NUM_HEADS * 16 * HEAD_DIM * 4 * NUM_LAYERS
    assert cache.memory_usage() == grown
    assert grown > initial


def test_constructor_validation():
    import pytest
    with pytest.raises(ValueError):
        DynamicKVCache(NUM_LAYERS, 1, 16, NUM_HEADS, HEAD_DIM, torch.float32, "cpu", initial_capacity=0)
    with pytest.raises(ValueError):
        DynamicKVCache(NUM_LAYERS, 1, 16, NUM_HEADS, HEAD_DIM, torch.float32, "cpu", growth_factor=1.0)
