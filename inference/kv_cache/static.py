"""Static KV cache.

Pre-allocates the full ``[B, H, max_seq_len, D]`` K/V tensors once at
construction and writes each new token block in place (no ``torch.cat``, no
per-step allocation). This is the classic "one allocation, one write per
token" strategy used by the first generation of inference engines.
"""

from __future__ import annotations

import torch

from inference.kv_cache.base import KVCache


class StaticKVCache(KVCache):
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

        shape = (max_batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        # Length per (layer, row): during one forward step every layer is
        # updated with the same positions, so a layer-shared length would make
        # layer 1 look like it appends after layer 0 already advanced.
        self.lengths = torch.zeros(
            (num_layers, max_batch_size), dtype=torch.long, device=self.device
        )
        self._allocation_count = 2 * num_layers

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
            # Append at each row's current length (rows may differ).
            for row in range(batch):
                start = int(self.lengths[layer_idx, row])
                if start + t_new > self.max_seq_len:
                    raise ValueError(
                        f"row {row}: sequence length {start + t_new} exceeds "
                        f"max_seq_len {self.max_seq_len}"
                    )
                self.k_cache[layer_idx][row, :, start:start + t_new, :] = key[row]
                self.v_cache[layer_idx][row, :, start:start + t_new, :] = value[row]
                self.lengths[layer_idx, row] = start + t_new
            return

        if int(positions[-1]) + 1 > self.max_seq_len:
            raise ValueError(
                f"sequence length {int(positions[-1]) + 1} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        # Same global positions for every row -> single advanced-indexing write.
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
        # The whole pre-allocated buffer counts, whether it is used or not.
        return self._tensor_bytes(self.k_cache) + self._tensor_bytes(self.v_cache)
