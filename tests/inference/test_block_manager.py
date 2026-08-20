"""Tests for the block manager (docs/kv_cache.md section 十一).

The critical property is the 申请 -> 释放 -> 再次申请 (allocate -> free ->
re-allocate) cycle: a freed block must be reusable by a later request.
"""

import pytest

from inference.kv_cache import Block, BlockManager

NUM_BLOCKS = 16
BLOCK_SIZE = 16


@pytest.fixture
def manager():
    return BlockManager(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)


def test_block_allocate(manager):
    blocks = manager.allocate(3)
    assert len(blocks) == 3
    # LIFO: the first allocation takes the highest free ids.
    assert sorted(b.block_id for b in blocks) == [NUM_BLOCKS - 3, NUM_BLOCKS - 2, NUM_BLOCKS - 1]
    assert all(b.ref_count == 1 for b in blocks)
    assert manager.num_allocated_blocks() == 3
    assert manager.num_free_blocks() == NUM_BLOCKS - 3
    for block in blocks:
        assert manager.has_block(block.block_id)


def test_block_allocate_initializes_ref_count(manager):
    block = manager.allocate(1)[0]
    assert block.ref_count == 1


def test_block_free_returns_to_free_list(manager):
    blocks = manager.allocate(2)
    manager.free(blocks)
    assert manager.num_allocated_blocks() == 0
    assert manager.num_free_blocks() == NUM_BLOCKS
    for block in blocks:
        assert block.ref_count == 0
        assert not manager.has_block(block.block_id)


def test_block_reuse_after_free(manager):
    """Free a block, allocate again, and the same physical id must come back."""
    first = manager.allocate(1)[0]
    manager.free([first])
    second = manager.allocate(1)[0]
    assert second.block_id == first.block_id


def test_allocate_free_reallocate_cycle(manager):
    """申请 -> 释放 -> 再次申请 must reuse the same physical blocks."""
    a = manager.allocate(4)
    ids_a = {b.block_id for b in a}
    manager.free(a)
    b = manager.allocate(4)
    ids_b = {b.block_id for b in b}
    assert ids_a == ids_b  # same physical blocks, reused after release


def test_allocate_multiple_requests(manager):
    a = manager.allocate(3)
    b = manager.allocate(2)
    assert manager.num_allocated_blocks() == 5
    assert manager.num_free_blocks() == NUM_BLOCKS - 5
    assert len({x.block_id for x in a + b}) == 5  # no overlap


def test_ref_count_decrements_one_per_free(manager):
    block = manager.allocate(1)[0]
    block.increment_ref()  # a second sequence shares the block
    assert block.ref_count == 2
    manager.free([block])  # one release -> still referenced
    assert block.ref_count == 1
    assert manager.has_block(block.block_id)
    manager.free([block])  # final release -> freed
    assert block.ref_count == 0
    assert not manager.has_block(block.block_id)
    assert manager.num_free_blocks() == NUM_BLOCKS


def test_out_of_blocks_raises(manager):
    manager.allocate(NUM_BLOCKS)
    with pytest.raises(RuntimeError, match="out of memory"):
        manager.allocate(1)


def test_allocate_more_than_capacity_raises(manager):
    with pytest.raises(RuntimeError, match="out of memory"):
        manager.allocate(NUM_BLOCKS + 1)


def test_free_unallocated_block_raises(manager):
    with pytest.raises(ValueError):
        manager.free([Block(block_id=99)])


def test_free_by_id(manager):
    blocks = manager.allocate(2)
    manager.free([b.block_id for b in blocks])
    assert manager.num_allocated_blocks() == 0


def test_blocks_handed_out_counts_reuse(manager):
    assert manager.blocks_handed_out == 0
    a = manager.allocate(2)
    manager.free(a)
    manager.allocate(2)
    assert manager.blocks_handed_out == 4


def test_constructor_validation():
    with pytest.raises(ValueError):
        BlockManager(num_blocks=0, block_size=16)
    with pytest.raises(ValueError):
        BlockManager(num_blocks=16, block_size=0)
