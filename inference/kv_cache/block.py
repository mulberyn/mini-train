"""A physical KV cache block.

In a paged cache the KV pool is sliced into fixed-size blocks of
``block_size`` tokens. A :class:`Block` is the bookkeeping object that tracks
one physical block: its id inside the pool and how many sequences currently
reference it (reference counting is what allows blocks to be freed and reused
as soon as the last sequence releases them).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Block:
    """One physical block of the KV pool.

    Attributes:
        block_id: index of this block inside the physical KV pool.
        ref_count: number of sequences that currently reference this block.
    """

    block_id: int
    ref_count: int = field(default=0)

    def increment_ref(self) -> None:
        self.ref_count += 1

    def decrement_ref(self) -> None:
        if self.ref_count <= 0:
            raise RuntimeError(
                f"cannot decrement ref_count of block {self.block_id} (already 0)"
            )
        self.ref_count -= 1
