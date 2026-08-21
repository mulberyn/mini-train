"""Block manager for the paged KV cache.

The block manager owns the bookkeeping of physical blocks:

* a free list (``free_blocks``) with the blocks that are currently unused;
* a map of allocated blocks (``allocated_blocks``) with their reference
  counts.

Allocating a block hands it out to a sequence (``ref_count`` starts at 1);
freeing a block decrements the reference count and returns the block to the
free list the moment it reaches zero, so it can be reused by a later request.
This is exactly the "申请 -> 释放 -> 再次申请" cycle that makes paged caching
memory-efficient.

The block manager is deliberately agnostic about the tensor layout of the KV
pool (``[num_blocks, block_size, num_kv_heads, head_dim]``); the
:class:`~inference.kv_cache.paged.PagedKVCache` owns the tensors and asks the
manager for block ids.
"""

from __future__ import annotations

from inference.kv_cache.block import Block


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # LIFO free list: recently freed blocks are reused first, which keeps
        # them hot in cache and makes the allocate/free/reallocate cycle
        # immediately reuse the same physical ids.
        self.free_blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]
        self.allocated_blocks: dict[int, Block] = {}
        self._blocks_handed_out = 0

    # ------------------------------------------------------------------ #
    def allocate(self, num_blocks: int = 1) -> list[Block]:
        """Allocate ``num_blocks`` physical blocks and return them.

        Each returned block has ``ref_count == 1`` (owned by its caller, e.g. a
        sequence's block table).

        Raises:
            RuntimeError: if fewer than ``num_blocks`` free blocks remain.
        """
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if num_blocks > len(self.free_blocks):
            raise RuntimeError(
                f"out of memory: requested {num_blocks} blocks, only "
                f"{len(self.free_blocks)} free (capacity {self.num_blocks})"
            )
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        for block in blocks:
            block.ref_count = 1
            self.allocated_blocks[block.block_id] = block
        self._blocks_handed_out += num_blocks
        return blocks

    # ------------------------------------------------------------------ #
    def free(self, blocks: list[Block] | list[int]) -> None:
        """Release blocks, returning them to the free list when unused.

        Decrements each block's reference count once; a block with
        ``ref_count == 0`` is returned to ``free_blocks`` and can be reused.
        """
        for item in blocks:
            if isinstance(item, Block):
                block = item
                if block.block_id not in self.allocated_blocks:
                    raise ValueError(f"block {block.block_id} is not allocated")
            else:
                block = self.allocated_blocks.get(item)
                if block is None:
                    raise ValueError(f"block {item} is not allocated")
            block.decrement_ref()
            if block.ref_count == 0:
                self.allocated_blocks.pop(block.block_id, None)
                self.free_blocks.append(block)

    # ------------------------------------------------------------------ #
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def num_allocated_blocks(self) -> int:
        return len(self.allocated_blocks)

    @property
    def blocks_handed_out(self) -> int:
        """Total number of blocks ever handed out (a reuse/alloc metric)."""
        return self._blocks_handed_out

    def has_block(self, block_id: int) -> bool:
        return block_id in self.allocated_blocks    

    def get_block(self, block_id: int) -> Block:
        return self.allocated_blocks[block_id]
