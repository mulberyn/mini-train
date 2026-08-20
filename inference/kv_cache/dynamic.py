"""Dynamic KV cache.

A middle ground between the naive and static caches: K/V tensors grow on
demand by doubling their capacity (like a dynamic array). The number of
allocations is ``O(log T)`` instead of ``O(T)`` (naive) while the memory used
is proportional to the *current* capacity rather than the worst case
``max_seq_len`` (static).
"""

from __future__ import annotations

import torch

from inference.kv_cache.base import KVCache


class DynamicKVCache(KVCache):
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        initial_capacity: int = 64,
        growth_factor: float = 2.0,
    ):
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if initial_capacity <= 0:
            raise ValueError(f"initial_capacity must be positive, got {initial_capacity}")
        if growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1.0, got {growth_factor}")
        self._validate_constructor_args(num_layers, max_batch_size, num_kv_heads, head_dim)
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.growth_factor = growth_factor

        capacity = min(max(initial_capacity, 1), max_seq_len)
        self.capacities = [capacity] * num_layers
        shape = (max_batch_size, num_kv_heads, capacity, head_dim)
        self.k_cache = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.lengths = torch.zeros(
            (num_layers, max_batch_size), dtype=torch.long, device=self.device
        )
        self._allocation_count = 2 * num_layers

    # ------------------------------------------------------------------ #
    def _ensure_capacity(self, layer_idx: int, needed: int) -> None:
        """Grow layer ``layer_idx`` until its capacity covers ``needed``."""
        while self.capacities[layer_idx] < needed:
            old_cap = self.capacities[layer_idx]
            new_cap = min(int(old_cap * self.growth_factor), self.max_seq_len)
            if new_cap <= old_cap:
                new_cap = needed
            new_k = torch.zeros(
                (self.max_batch_size, self.num_kv_heads, new_cap, self.head_dim),
                dtype=self.dtype, device=self.device,
            )
            new_v = torch.zeros_like(new_k)
            used = min(old_cap, int(self.lengths[layer_idx].max().item()))
            new_k[:, :, :used, :] = self.k_cache[layer_idx][:, :, :used, :]
            new_v[:, :, :used, :] = self.v_cache[layer_idx][:, :, :used, :]
            self.k_cache[layer_idx] = new_k
            self.v_cache[layer_idx] = new_v
            self.capacities[layer_idx] = new_cap
            self._allocation_count += 2

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

        if positions is None:
            for row in range(batch):
                start = int(self.lengths[layer_idx, row])
                self._ensure_capacity(layer_idx, start + t_new)
                self.k_cache[layer_idx][row, :, start:start + t_new, :] = key[row]
                self.v_cache[layer_idx][row, :, start:start + t_new, :] = value[row]
                self.lengths[layer_idx, row] = start + t_new
            return

        if int(positions[-1]) + 1 > self.max_seq_len:
            raise ValueError(
                f"sequence length {int(positions[-1]) + 1} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        self._ensure_capacity(layer_idx, int(positions[-1]) + 1)
        self.k_cache[layer_idx][:, :, positions, :] = key
        self.v_cache[layer_idx][:, :, positions, :] = value
        self.lengths[layer_idx, :batch] = torch.maximum(
            self.lengths[layer_idx, :batch],
            torch.full_like(self.lengths[layer_idx, :batch], int(positions[-1]) + 1),
        )

    # ------------------------------------------------------------------ #
    def get(
        self,
        layer_idx: int,
        positions: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        if positions is None:
            length = int(self.lengths[layer_idx].max().item())
            k = self.k_cache[layer_idx][:, :, :length, :]
            v = self.v_cache[layer_idx][:, :, :length, :]
            return k, v
        if isinstance(positions, torch.Tensor):
            positions = positions.to(self.device)
            k = self.k_cache[layer_idx][:, :, positions, :]
            v = self.v_cache[layer_idx][:, :, positions, :]
            return k, v
        k = self.k_cache[layer_idx][:, :, positions:positions + 1, :]
        v = self.v_cache[layer_idx][:, :, positions:positions + 1, :]
        return k, v

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        for layer_idx in range(self.num_layers):
            self.k_cache[layer_idx].zero_()
            self.v_cache[layer_idx].zero_()
        self.lengths.zero_()

    # ------------------------------------------------------------------ #
    def memory_usage(self) -> float:
        # Count the current capacity (not the worst-case max_seq_len).
        total = 0.0
        for layer_idx in range(self.num_layers):
            total += self.k_cache[layer_idx].numel() * self.k_cache[layer_idx].element_size()
            total += self.v_cache[layer_idx].numel() * self.v_cache[layer_idx].element_size()
        return total
