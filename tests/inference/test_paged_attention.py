"""Tests for the paged attention reference implementation.

The core property under test (docs/kv_cache.md section 十三):

    Paged Attention == 普通 Attention

i.e. gathering K/V through the block table and computing attention must
produce exactly the same result as dense attention over the gathered K/V.
"""

import math

import pytest
import torch

from inference.attention import dense_attention, paged_attention, paged_attention_from_cache
from inference.kv_cache.paged import PagedKVCache

NUM_BLOCKS = 64
BLOCK_SIZE = 8
NUM_KV_HEADS = 4
HEAD_DIM = 16


@pytest.fixture
def pools():
    torch.manual_seed(7)
    key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    value_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return key_cache, value_cache


def dense_reference(query, key_cache, value_cache, block_tables, context_lengths, block_size):
    """Gather dense K/V via the block table, then run dense attention."""
    batch, num_heads, head_dim = query.shape
    scale = 1.0 / math.sqrt(head_dim)
    outs = []
    for b in range(batch):
        ctx = int(context_lengths[b])
        n_blocks = (ctx + block_size - 1) // block_size
        k_parts, v_parts = [], []
        for logical in range(n_blocks):
            physical = int(block_tables[b, logical])
            k = key_cache[physical]
            v = value_cache[physical]
            if logical == n_blocks - 1:
                rem = ctx - logical * block_size
                k = k[:rem]
                v = v[:rem]
            k_parts.append(k)
            v_parts.append(v)
        k = torch.cat(k_parts, dim=0).transpose(0, 1).unsqueeze(0)  # [1, H, T, D]
        v = torch.cat(v_parts, dim=0).transpose(0, 1).unsqueeze(0)
        q = query[b:b + 1].unsqueeze(2)  # [1, H, 1, D]
        outs.append(dense_attention(q, k, v, scale=scale))
    return torch.cat(outs, dim=0).squeeze(2)


def test_matches_dense_single_query(pools):
    key_cache, value_cache = pools
    ctx = 13
    block_tables = torch.randint(0, NUM_BLOCKS, (1, 2))
    context_lengths = torch.tensor([ctx])
    query = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)

    actual = paged_attention(query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)
    expected = dense_reference(query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("context_length", [1, 7, 8, 9, 16, 17, 31, 64])
def test_matches_dense_various_lengths(pools, context_length):
    """Context lengths that are not block-aligned must still match."""
    key_cache, value_cache = pools
    batch = 3
    n_blocks = (context_length + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (batch, n_blocks))
    context_lengths = torch.full((batch,), context_length, dtype=torch.long)
    query = torch.randn(batch, NUM_KV_HEADS, HEAD_DIM)

    actual = paged_attention(query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)
    expected = dense_reference(query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_matches_dense_mixed_lengths(pools):
    """Different sequences may have different context lengths (paged style)."""
    key_cache, value_cache = pools
    batch = 4
    context_lengths = torch.tensor([5, 8, 17, 32])
    max_blocks = 4
    block_tables = torch.randint(0, NUM_BLOCKS, (batch, max_blocks))
    query = torch.randn(batch, NUM_KV_HEADS, HEAD_DIM)

    actual = paged_attention(query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)
    expected = dense_reference(query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_matches_dense_repeated_random_configs(pools):
    """Fuzz: many random layouts must all match the dense reference."""
    key_cache, value_cache = pools
    for _ in range(20):
        batch = torch.randint(1, 5, ()).item()
        ctx = int(torch.randint(1, 40, ()).item())
        n_blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_tables = torch.randint(0, NUM_BLOCKS, (batch, n_blocks))
        context_lengths = torch.full((batch,), ctx, dtype=torch.long)
        query = torch.randn(batch, NUM_KV_HEADS, HEAD_DIM)

        actual = paged_attention(query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)
        expected = dense_reference(query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_gqa_repeats_kv_heads(pools):
    """GQA: 8 query heads over 4 KV heads must match a dense GQA reference."""
    key_cache, value_cache = pools
    num_heads, num_kv_heads = 8, 4
    ctx = 17
    block_tables = torch.randint(0, NUM_BLOCKS, (2, 3))
    context_lengths = torch.tensor([17, 17])
    query = torch.randn(2, num_heads, HEAD_DIM)

    actual = paged_attention(
        query, key_cache, value_cache, block_tables, context_lengths,
        BLOCK_SIZE, num_kv_heads=num_kv_heads,
    )
    # Dense reference with GQA head grouping.
    outs = []
    for b in range(2):
        n_blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        k_parts, v_parts = [], []
        for logical in range(n_blocks):
            physical = int(block_tables[b, logical])
            k = key_cache[physical]
            v = value_cache[physical]
            if logical == n_blocks - 1:
                rem = ctx - logical * BLOCK_SIZE
                k = k[:rem]
                v = v[:rem]
            k_parts.append(k)
            v_parts.append(v)
        k = torch.cat(k_parts, dim=0)  # [T, kv, D]
        v = torch.cat(v_parts, dim=0)
        kv_index = torch.arange(num_heads) // (num_heads // num_kv_heads)
        k = k[:, kv_index, :].transpose(0, 1).unsqueeze(0)  # [1, H, T, D]
        v = v[:, kv_index, :].transpose(0, 1).unsqueeze(0)
        q = query[b:b + 1].unsqueeze(2)
        outs.append(dense_attention(q, k, v, scale=1.0 / math.sqrt(HEAD_DIM)))
    expected = torch.cat(outs, dim=0).squeeze(2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_paged_attention_from_cache():
    """End-to-end: fill a PagedKVCache, then run paged attention on it."""
    num_layers, batch, num_blocks = 2, 1, 32
    cache = PagedKVCache(
        num_layers=num_layers, max_batch_size=batch, num_blocks=num_blocks,
        block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        dtype=torch.float32, device="cpu",
    )
    cache.allocate_sequence(0)
    torch.manual_seed(3)
    num_tokens = 13
    for t in range(num_tokens):
        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        cache.update(0, k, v, torch.tensor([t]))
        cache.update(1, k, v, torch.tensor([t]))

    query = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    actual = paged_attention_from_cache(query, cache, 0, layer_idx=0)

    # Dense reference from the cache's own get() view.
    k, v = cache.get(0)
    q = query.unsqueeze(2)
    expected = dense_attention(q, k, v, scale=1.0 / math.sqrt(HEAD_DIM)).squeeze(2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_paged_attention_unknown_sequence_raises(pools):
    key_cache, value_cache = pools
    cache = PagedKVCache(
        num_layers=1, max_batch_size=2, num_blocks=16, block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, dtype=torch.float32, device="cpu",
    )
    cache.allocate_sequence(0)
    query = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    with pytest.raises(ValueError):
        paged_attention_from_cache(query, cache, 5)


def test_paged_attention_rejects_bad_shapes(pools):
    key_cache, value_cache = pools
    block_tables = torch.randint(0, NUM_BLOCKS, (2, 2))
    context_lengths = torch.tensor([5, 5])
    query = torch.randn(2, NUM_KV_HEADS, HEAD_DIM)
    with pytest.raises(ValueError):
        paged_attention(query, key_cache, value_cache, block_tables, context_lengths, 0)
    with pytest.raises(ValueError):
        paged_attention(query[0], key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)
    with pytest.raises(ValueError):
        paged_attention(query, key_cache, value_cache, block_tables, torch.tensor([5]), BLOCK_SIZE)


def test_paged_attention_raises_on_unallocated_block(pools):
    key_cache, value_cache = pools
    block_tables = torch.tensor([[-1, -1]])
    context_lengths = torch.tensor([5])
    query = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    with pytest.raises(RuntimeError):
        paged_attention(query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)


def test_paged_attention_empty_context_returns_zero(pools):
    key_cache, value_cache = pools
    block_tables = torch.zeros(1, 1, dtype=torch.long)
    context_lengths = torch.tensor([0])
    query = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
    actual = paged_attention(query, key_cache, value_cache, block_tables, context_lengths, BLOCK_SIZE)
    torch.testing.assert_close(actual, torch.zeros_like(query))
