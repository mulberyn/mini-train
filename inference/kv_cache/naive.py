"""Naive KV cache.

The simplest possible cache: every :meth:`update` concatenates the new keys and
values onto the previous tensors with ``torch.cat``, so each decode step
allocates a brand-new ``[B, H, T+1, D]`` tensor and copies the whole history.

This is the reference implementation that every other cache is measured
against. It is correct but wasteful: ``O(T)`` allocations and ``O(T^2)``
copies over a generation of ``T`` tokens.
"""

from __future__ import annotations

import torch

from inference.kv_cache.base import KVCache


class NaiveKVCache(KVCache):
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self._validate_constructor_args(num_layers, max_batch_size, num_kv_heads, head_dim)
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        # K/V per layer, each a growing [B, H, T, D] tensor (None until first use).
        self.k_cache: list[torch.Tensor | None] = [None] * num_layers
        self.v_cache: list[torch.Tensor | None] = [None] * num_layers
        # Length is tracked per layer: every layer is updated with the same
        # positions during one forward pass, and the check must not see the
        # length bumped by the previous layer.
        self.lengths: list[int] = [0] * num_layers
        self._allocation_count = 0

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
        positions = self._resolve_positions(positions, t_new, self.lengths[layer_idx], self.device)
        if int(positions[0]) != self.lengths[layer_idx]:
            raise ValueError(
                f"NaiveKVCache appends sequentially; expected positions to start "
                f"at {self.lengths[layer_idx]}, got {int(positions[0])}"
            )
        if int(positions[-1]) + 1 > self.max_seq_len:
            raise ValueError(
                f"sequence length {int(positions[-1]) + 1} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )

        old_k = self.k_cache[layer_idx]
        old_v = self.v_cache[layer_idx]
        self.k_cache[layer_idx] = key if old_k is None else torch.cat([old_k, key], dim=-2)
        self.v_cache[layer_idx] = value if old_v is None else torch.cat([old_v, value], dim=-2)
        self._allocation_count += 2
        self.lengths[layer_idx] = int(positions[-1]) + 1

    # ------------------------------------------------------------------ #
    def get(
        self,
        layer_idx: int,
        positions: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        k = self.k_cache[layer_idx]
        v = self.v_cache[layer_idx]
        if k is None:
            empty = torch.empty(
                self.max_batch_size, self.num_kv_heads, 0, self.head_dim,
                dtype=self.dtype, device=self.device,
            )
            return empty, empty.clone()
        if positions is not None:
            if isinstance(positions, torch.Tensor):
                positions = positions.to(self.device)
                k = k[:, :, positions, :]
                v = v[:, :, positions, :]
            else:
                k = k[:, :, positions:positions + 1, :]
                v = v[:, :, positions:positions + 1, :]
        return k, v

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers
        self.lengths = [0] * self.num_layers

    # ------------------------------------------------------------------ #
    def memory_usage(self) -> float:
        return self._tensor_bytes(self.k_cache) + self._tensor_bytes(self.v_cache)
