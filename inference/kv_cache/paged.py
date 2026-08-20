"""Paged KV cache.

The paged cache slices the physical KV pool into fixed-size blocks and gives
each sequence a *block table* mapping its logical blocks to physical blocks:

    logical block 0 -> physical block 17
    logical block 1 -> physical block 4
    ...

Memory is only consumed by blocks that are actually allocated to live
sequences, and a block is freed (and can be reused by another sequence) as
soon as its last reference disappears. This is the strategy used by vLLM /
SGLang and it is what enables high-concurrency serving without reserving
``max_seq_len`` per request.

Tensor layout
-------------
For every layer the pool is a single tensor of shape
``[num_blocks, block_size, num_kv_heads, head_dim]`` for K and one for V.
The :class:`~inference.kv_cache.block_manager.BlockManager` owns the
bookkeeping (free / allocated / ref counts); this class owns the tensors and
the per-sequence state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from inference.kv_cache.base import KVCache
from inference.kv_cache.block_manager import BlockManager


@dataclass
class SequenceState:
    """Mutable state of one in-flight sequence inside the paged cache.

    The token count is tracked *per layer*: during one model forward step every
    layer is updated with the same positions, so a single shared length would
    make layer 1 look like it is appending after layer 0 already advanced the
    sequence.
    """

    seq_id: int
    block_table: list[int] = field(default_factory=list)  # physical block ids
    _lengths: dict[int, int] = field(default_factory=dict)

    def length_for(self, layer_idx: int) -> int:
        return self._lengths.get(layer_idx, 0)

    def set_length(self, layer_idx: int, length: int) -> None:
        self._lengths[layer_idx] = length

    @property
    def length(self) -> int:
        """Sequence length (tokens), i.e. the max over all layers."""
        return max(self._lengths.values(), default=0)


class PagedKVCache(KVCache):
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self._validate_constructor_args(num_layers, max_batch_size, num_kv_heads, head_dim)
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        self.block_manager = BlockManager(num_blocks, block_size)
        pool_shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k_pool = [
            torch.zeros(pool_shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.v_pool = [
            torch.zeros(pool_shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        # Sequences are keyed by *batch row* (0 .. max_batch_size-1), matching
        # the row indexing of the KVCache.update(key[..., B, ...]) interface.
        self.sequences: dict[int, SequenceState] = {}
        self._allocation_count = 2 * num_layers

    # ------------------------------------------------------------------ #
    # Sequence lifecycle
    # ------------------------------------------------------------------ #
    def allocate_sequence(self, seq_id: int | None = None) -> int:
        """Allocate a sequence slot and return its id (the batch row).

        Args:
            seq_id: explicit batch row to use; defaults to the lowest free row.
        """
        if seq_id is None:
            for candidate in range(self.max_batch_size):
                if candidate not in self.sequences:
                    seq_id = candidate
                    break
            if seq_id is None:
                raise RuntimeError(f"paged cache is full ({self.max_batch_size} sequences)")
        if not 0 <= seq_id < self.max_batch_size:
            raise ValueError(f"seq_id {seq_id} out of range [0, {self.max_batch_size})")
        if seq_id in self.sequences:
            raise ValueError(f"sequence {seq_id} is already allocated")
        self.sequences[seq_id] = SequenceState(seq_id=seq_id)
        return seq_id

    def release_sequence(self, seq_id: int) -> None:
        """Release a sequence, freeing every block in its block table."""
        if seq_id not in self.sequences:
            raise ValueError(f"sequence {seq_id} is not allocated")
        seq = self.sequences.pop(seq_id)
        if seq.block_table:
            self.block_manager.free(seq.block_table)

    # ------------------------------------------------------------------ #
    # Block table helpers
    # ------------------------------------------------------------------ #
    def _ensure_blocks(self, seq: SequenceState, num_blocks_needed: int) -> None:
        while len(seq.block_table) < num_blocks_needed:
            blocks = self.block_manager.allocate(1)
            seq.block_table.append(blocks[0].block_id)

    def get_block_tables(self) -> torch.Tensor:
        """Return per-row block tables as ``[max_batch_size, max_num_blocks]``.

        Rows without a sequence (or without blocks) are padded with ``-1``.
        """
        max_blocks = 0
        for seq in self.sequences.values():
            max_blocks = max(max_blocks, len(seq.block_table))
        tables = torch.full(
            (self.max_batch_size, max_blocks), -1, dtype=torch.long, device=self.device
        )
        for row, seq in self.sequences.items():
            for j, phys in enumerate(seq.block_table):
                tables[row, j] = phys
        return tables

    def get_context_lengths(self) -> torch.Tensor:
        """Return per-row cached token counts as ``[max_batch_size]`` longs."""
        lengths = torch.zeros(self.max_batch_size, dtype=torch.long, device=self.device)
        for row, seq in self.sequences.items():
            lengths[row] = seq.length
        return lengths

    # ------------------------------------------------------------------ #
    # KVCache interface
    # ------------------------------------------------------------------ #
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> None:
        batch, t_new = self._validate_update_inputs(layer_idx, key, value, positions)
        if t_new == 0:
            return
        for row in range(batch):
            if row not in self.sequences:
                raise RuntimeError(
                    f"batch row {row} has no allocated sequence; call "
                    f"allocate_sequence({row}) first"
                )
        for row in range(batch):
            seq = self.sequences[row]
            if positions is None:
                row_positions = torch.arange(
                    seq.length_for(layer_idx),
                    seq.length_for(layer_idx) + t_new,
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                row_positions = positions
                if int(row_positions[0]) != seq.length_for(layer_idx):
                    raise ValueError(
                        f"row {row}: update positions must start at current length "
                        f"{seq.length_for(layer_idx)}, got {int(row_positions[0])}"
                    )
            self._append_row(layer_idx, seq, key[row], value[row], row_positions)

    def _append_row(
        self,
        layer_idx: int,
        seq: SequenceState,
        k_row: torch.Tensor,
        v_row: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Append one batch row's K/V (``[H, T_new, D]``) at ``positions``."""
        t_new = k_row.shape[1]
        for t in range(t_new):
            pos = int(positions[t])
            block_idx = pos // self.block_size
            slot = pos % self.block_size
            self._ensure_blocks(seq, block_idx + 1)
            phys = seq.block_table[block_idx]
            self.k_pool[layer_idx][phys, slot, :, :] = k_row[:, t, :]
            self.v_pool[layer_idx][phys, slot, :, :] = v_row[:, t, :]
        seq.set_length(layer_idx, max(seq.length_for(layer_idx), int(positions[-1]) + 1))

    def get(
        self,
        layer_idx: int,
        positions: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

        if positions is None:
            t_out = max((seq.length for seq in self.sequences.values()), default=0)
        elif isinstance(positions, torch.Tensor):
            t_out = positions.numel()
        else:
            t_out = 1

        out_k = torch.zeros(
            (self.max_batch_size, self.num_kv_heads, t_out, self.head_dim),
            dtype=self.dtype, device=self.device,
        )
        out_v = torch.zeros_like(out_k)

        for row, seq in self.sequences.items():
            if positions is None:
                for pos in range(seq.length):
                    block_idx = pos // self.block_size
                    slot = pos % self.block_size
                    phys = seq.block_table[block_idx]
                    out_k[row, :, pos, :] = self.k_pool[layer_idx][phys, slot, :, :]
                    out_v[row, :, pos, :] = self.v_pool[layer_idx][phys, slot, :, :]
            else:
                pos_list = (
                    positions.to(self.device).tolist()
                    if isinstance(positions, torch.Tensor)
                    else [positions]
                )
                for j, pos in enumerate(pos_list):
                    if pos >= seq.length:
                        raise IndexError(
                            f"row {row}: position {pos} beyond cached length {seq.length}"
                        )
                    block_idx = pos // self.block_size
                    slot = pos % self.block_size
                    phys = seq.block_table[block_idx]
                    out_k[row, :, j, :] = self.k_pool[layer_idx][phys, slot, :, :]
                    out_v[row, :, j, :] = self.v_pool[layer_idx][phys, slot, :, :]
        return out_k, out_v

    def reset(self) -> None:
        for seq in self.sequences.values():
            if seq.block_table:
                self.block_manager.free(seq.block_table)
        self.sequences.clear()

    def memory_usage(self) -> float:
        """Bytes occupied by K and V of the *allocated* blocks only."""
        block_bytes = self.block_size * self.num_kv_heads * self.head_dim
        block_bytes *= self._dtype_size(self.dtype)
        allocated = self.block_manager.num_allocated_blocks()
        return float(allocated * block_bytes * 2 * self.num_layers)

    # ------------------------------------------------------------------ #
    # Paged-specific introspection
    # ------------------------------------------------------------------ #
    @property
    def pool_bytes(self) -> float:
        """Bytes of the whole physical pool (allocated + free blocks)."""
        block_bytes = self.block_size * self.num_kv_heads * self.head_dim
        block_bytes *= self._dtype_size(self.dtype)
        return float(self.num_blocks * block_bytes * 2 * self.num_layers)

    def num_allocated_blocks(self) -> int:
        return self.block_manager.num_allocated_blocks()

    def num_free_blocks(self) -> int:
        return self.block_manager.num_free_blocks()
