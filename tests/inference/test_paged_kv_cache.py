"""Tests for the paged KV cache (docs/kv_cache.md sections 八 / 九 / 十一)."""

import pytest
import torch

from inference.kv_cache import BlockManager, PagedKVCache
from inference.kv_cache.paged import SequenceState
from tests.inference.conftest import (
    CONTEXT_LENGTH,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    make_cache,
    update_tokens,
)

HEAD_DIM = D_MODEL // NUM_HEADS
BLOCK_SIZE = 4


def make_paged(**kwargs):
    return make_cache("paged", block_size=BLOCK_SIZE, **kwargs)


def test_sequence_allocate():
    cache = make_paged(max_batch_size=4, num_blocks=16)
    seq_id = cache.allocate_sequence()
    assert seq_id == 0
    seq_id2 = cache.allocate_sequence()
    assert seq_id2 == 1
    assert len(cache.sequences) == 2
    assert cache.sequences[0].block_table == []
    assert cache.sequences[0].length == 0


def test_sequence_allocate_explicit_id():
    cache = make_paged(max_batch_size=4, num_blocks=16)
    cache.allocate_sequence(2)
    assert 2 in cache.sequences
    with pytest.raises(ValueError):
        cache.allocate_sequence(2)  # duplicate
    with pytest.raises(ValueError):
        cache.allocate_sequence(99)  # out of range


def test_sequence_release_frees_blocks():
    cache = make_paged(max_batch_size=2, num_blocks=16)
    cache.allocate_sequence(0)
    cache.allocate_sequence(1)
    key = torch.randn(2, NUM_HEADS, 5, HEAD_DIM)
    update_tokens(cache, key, key, torch.arange(5))
    assert cache.num_allocated_blocks() == 2 * 2  # 2 sequences * 2 blocks (ceil(5/4))
    cache.release_sequence(0)
    assert 0 not in cache.sequences
    assert cache.num_allocated_blocks() == 2  # only sequence 1 remains
    assert cache.num_free_blocks() == 16 - 2


def test_block_table_logical_to_physical():
    """Appending tokens must grow the block table with physical block ids."""
    cache = make_paged(max_batch_size=1, num_blocks=16)
    cache.allocate_sequence(0)
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    for t in range(9):  # ceil(9/4) = 3 blocks
        update_tokens(cache, key, key, torch.tensor([t]))
    seq = cache.sequences[0]
    assert len(seq.block_table) == 3
    assert all(phys >= 0 for phys in seq.block_table)
    assert len(set(seq.block_table)) == 3  # distinct physical blocks
    tables = cache.get_block_tables()
    assert tables.shape == (1, 3)
    assert tables[0].tolist() == seq.block_table


def test_append_token_places_tokens_in_blocks():
    cache = make_paged(max_batch_size=1, num_blocks=16)
    cache.allocate_sequence(0)
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    update_tokens(cache, key, key, torch.tensor([0]))
    # Token 0 must live in block slot 0 of the first physical block.
    seq = cache.sequences[0]
    phys = seq.block_table[0]
    torch.testing.assert_close(cache.k_pool[0][phys, 0], key[0, :, 0, :])
    # Tokens 1..4 (4 tokens): token 4 is the first token of block 1, slot 0.
    key4 = torch.randn(1, NUM_HEADS, 4, HEAD_DIM)
    update_tokens(cache, key4, key4, torch.arange(1, 5))
    seq = cache.sequences[0]
    assert len(seq.block_table) == 2
    phys = seq.block_table[1]
    torch.testing.assert_close(cache.k_pool[0][phys, 0], key4[0, :, 3, :])
    # The remaining slots of the last partial block are untouched (zeros).
    torch.testing.assert_close(
        cache.k_pool[0][phys, 1:], torch.zeros_like(cache.k_pool[0][phys, 1:])
    )
    # Token 1 went to block 0 slot 1.
    phys0 = seq.block_table[0]
    torch.testing.assert_close(cache.k_pool[0][phys0, 1], key4[0, :, 0, :])


def test_multiple_sequences_are_isolated():
    cache = make_paged(max_batch_size=2, num_blocks=16)
    cache.allocate_sequence(0)
    cache.allocate_sequence(1)
    key = torch.randn(2, NUM_HEADS, 4, HEAD_DIM)  # row 0 and row 1 differ
    value = torch.randn(2, NUM_HEADS, 4, HEAD_DIM)
    cache.update(0, key, value, torch.arange(4))
    cache.update(1, key, value, torch.arange(4))
    k0, v0 = cache.get(0)
    k1, v1 = cache.get(1)
    torch.testing.assert_close(k0[0], key[0])
    torch.testing.assert_close(k0[1], key[1])
    torch.testing.assert_close(k1[0], key[0])
    torch.testing.assert_close(k1[1], key[1])
    torch.testing.assert_close(v1[1], value[1])
    torch.testing.assert_close(v0[0], value[0])
    assert k0.shape == (2, NUM_HEADS, 4, HEAD_DIM)


def test_get_matches_static_for_same_tokens():
    """Paged and static caches must store identical K/V content."""
    from inference.kv_cache import StaticKVCache

    paged = make_paged(max_batch_size=2, num_blocks=32)
    paged.allocate_sequence(0)
    paged.allocate_sequence(1)
    static = StaticKVCache(
        num_layers=NUM_LAYERS, max_batch_size=2, max_seq_len=16,
        num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM, dtype=torch.float32, device="cpu",
    )
    torch.manual_seed(0)
    for t in range(9):
        key = torch.randn(2, NUM_HEADS, 1, HEAD_DIM)
        value = torch.randn(2, NUM_HEADS, 1, HEAD_DIM)
        update_tokens(paged, key, value, torch.tensor([t]))
        update_tokens(static, key, value, torch.tensor([t]))
    for layer_idx in range(NUM_LAYERS):
        kp, vp = paged.get(layer_idx)
        ks, vs = static.get(layer_idx)
        torch.testing.assert_close(kp, ks)
        torch.testing.assert_close(vp, vs)


def test_memory_usage_tracks_allocated_blocks():
    cache = make_paged(max_batch_size=1, num_blocks=16)
    assert cache.memory_usage() == 0.0
    cache.allocate_sequence(0)
    block_bytes = (
        BLOCK_SIZE * NUM_HEADS * HEAD_DIM * 4 * 2 * NUM_LAYERS
    )
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    for t in range(BLOCK_SIZE):
        update_tokens(cache, key, key, torch.tensor([t]))
    assert cache.memory_usage() == block_bytes
    for t in range(BLOCK_SIZE, 2 * BLOCK_SIZE):
        update_tokens(cache, key, key, torch.tensor([t]))
    assert cache.memory_usage() == 2 * block_bytes


def test_pool_bytes_is_full_capacity():
    cache = make_paged(max_batch_size=1, num_blocks=16)
    expected = (
        16 * BLOCK_SIZE * NUM_HEADS * HEAD_DIM * 4 * 2 * NUM_LAYERS
    )
    assert cache.pool_bytes == expected


def test_cache_reset_frees_all_blocks():
    cache = make_paged(max_batch_size=2, num_blocks=16)
    cache.allocate_sequence(0)
    cache.allocate_sequence(1)
    key = torch.randn(2, NUM_HEADS, 5, HEAD_DIM)
    update_tokens(cache, key, key, torch.arange(5))
    assert cache.num_allocated_blocks() > 0
    cache.reset()
    assert len(cache.sequences) == 0
    assert cache.num_allocated_blocks() == 0
    assert cache.num_free_blocks() == 16
    assert cache.memory_usage() == 0.0
    k, _ = cache.get(0)
    assert k.shape[-2] == 0


def test_block_reuse_after_sequence_release():
    """Release a sequence, allocate a new one: physical blocks must be reused.

    The pool is sized *exactly* for one 9-token sequence (3 blocks). If the
    released blocks were not returned to the free list, the re-allocation
    would run out of memory -- so a successful re-allocate proves reuse.
    """
    cache = make_paged(max_batch_size=1, num_blocks=3)
    cache.allocate_sequence(0)
    key = torch.randn(1, NUM_HEADS, 9, HEAD_DIM)
    update_tokens(cache, key, key, torch.arange(9))
    assert cache.num_allocated_blocks() == 3
    assert cache.num_free_blocks() == 0

    cache.release_sequence(0)
    assert cache.num_free_blocks() == 3

    cache.allocate_sequence(0)
    update_tokens(cache, key, key, torch.arange(9))  # raises OOM if not reused
    assert cache.num_allocated_blocks() == 3
    assert cache.num_free_blocks() == 0


def test_update_without_sequence_raises():
    cache = make_paged(max_batch_size=1, num_blocks=16)
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    with pytest.raises(RuntimeError, match="allocate_sequence"):
        cache.update(0, key, key, torch.tensor([0]))


def test_update_positions_must_start_at_length():
    cache = make_paged(max_batch_size=1, num_blocks=16)
    cache.allocate_sequence(0)
    key = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
    update_tokens(cache, key, key, torch.tensor([0]))
    with pytest.raises(ValueError):
        cache.update(0, key, key, torch.tensor([2]))  # gap


def test_context_lengths():
    cache = make_paged(max_batch_size=2, num_blocks=16)
    cache.allocate_sequence(0)
    cache.allocate_sequence(1)
    key = torch.randn(2, NUM_HEADS, 3, HEAD_DIM)
    update_tokens(cache, key, key, torch.arange(3))
    lengths = cache.get_context_lengths()
    assert lengths.tolist() == [3, 3]


def test_get_positions_from_paged_cache():
    cache = make_paged(max_batch_size=1, num_blocks=16)
    cache.allocate_sequence(0)
    key = torch.randn(1, NUM_HEADS, 6, HEAD_DIM)
    update_tokens(cache, key, key, torch.arange(6))
    k, _ = cache.get(0, torch.tensor([1, 5]))
    torch.testing.assert_close(k[:, :, 0], key[:, :, 1])
    torch.testing.assert_close(k[:, :, 1], key[:, :, 5])
    with pytest.raises(IndexError):
        cache.get(0, 9)


def test_block_manager_wired_into_cache():
    cache = make_paged(max_batch_size=1, num_blocks=16)
    assert isinstance(cache.block_manager, BlockManager)
    assert cache.block_manager.block_size == BLOCK_SIZE


def test_sequence_state_helpers():
    seq = SequenceState(seq_id=3)
    assert seq.length == 0
    seq.set_length(0, 8)
    seq.set_length(1, 8)
    assert seq.length == 8
    assert seq.length_for(0) == 8
    assert seq.length_for(5) == 0
