"""Unified KV cache interface for the inference engine.

All KV cache implementations (naive / static / dynamic / paged) share this
interface so that the model runner, attention modules and benchmarks never
have to know which strategy is in use:

    update(layer_idx, key, value, positions)   # append / write K and V
    get(layer_idx, positions=None)             # (K, V) dense view [B, H, T, D]
    reset()                                    # drop all cached state
    memory_usage()                             # bytes currently occupied
    allocation_count                           # K/V tensor allocations so far

Tensor conventions
------------------
* ``key`` / ``value`` passed to :meth:`update` have shape
  ``[B, H, T_new, D]`` (``H`` == ``num_kv_heads`` for MHA; GQA is left for a
  later phase).
* ``positions`` is a 1-D tensor of global token positions ``[T_new]`` that is
  shared by every batch row (static batching: all rows advance in lock-step).
  During prefill it is ``arange(0, T_prompt)``; during decode step ``i`` it is
  ``[T_prompt + i]``.
* :meth:`get` returns ``(K, V)`` tensors of shape ``[B, H, T, D]`` where ``T``
  is the longest cached sequence (shorter rows are zero-padded).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class KVCache(ABC):
    """Abstract base class for every KV cache implementation."""

    num_layers: int
    max_batch_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    # ------------------------------------------------------------------ #
    # Required API
    # ------------------------------------------------------------------ #
    @abstractmethod
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> None:
        """Write ``key``/``value`` for layer ``layer_idx``.

        Args:
            layer_idx: Transformer layer index (``0 <= layer_idx < num_layers``).
            key: ``[B, H, T_new, D]`` key tensor for the new tokens.
            value: ``[B, H, T_new, D]`` value tensor for the new tokens.
            positions: global token positions ``[T_new]`` shared by all rows.
                ``None`` means "append at the current length" (``T_new`` must
                then be equal for every row).
        """

    @abstractmethod
    def get(
        self,
        layer_idx: int,
        positions: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached ``(K, V)`` tensors for a layer.

        Args:
            layer_idx: Transformer layer index.
            positions: optional ``[T]`` tensor (or int) of token positions to
                gather for every row; ``None`` returns the full cached prefix.

        Returns:
            ``(K, V)`` with shape ``[B, H, T, D]``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Drop all cached key/value state (cache can be reused afterwards)."""

    @abstractmethod
    def memory_usage(self) -> float:
        """Return the bytes currently occupied by K and V (all layers)."""

    # ------------------------------------------------------------------ #
    # Introspection shared by all implementations
    # ------------------------------------------------------------------ #
    @property
    def allocation_count(self) -> int:
        """Number of K/V tensor allocations performed since construction.

        This is the metric that separates the strategies: naive allocates two
        tensors per update, static allocates once up front, dynamic allocates
        only when the capacity doubles, and paged allocates the block pool once.
        """
        return self._allocation_count

    # ------------------------------------------------------------------ #
    # Shared validation helpers
    # ------------------------------------------------------------------ #
    def _validate_constructor_args(
        self,
        num_layers: int,
        max_batch_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

    def _validate_update_inputs(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[int, int]:
        """Validate an :meth:`update` call; return ``(B, T_new)``."""
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("key and value must be torch.Tensor")
        if key.shape != value.shape:
            raise ValueError(
                f"key {tuple(key.shape)} and value {tuple(value.shape)} must match"
            )
        if key.ndim != 4:
            raise ValueError(
                f"key/value must have shape [B, H, T, D], got {tuple(key.shape)}"
            )
        if key.size(0) > self.max_batch_size:
            raise ValueError(
                f"batch {key.size(0)} exceeds max_batch_size {self.max_batch_size}"
            )
        if key.size(1) != self.num_kv_heads:
            raise ValueError(
                f"key/value heads {key.size(1)} != num_kv_heads "
                f"{self.num_kv_heads} (GQA is not supported yet)"
            )
        if key.size(3) != self.head_dim:
            raise ValueError(
                f"key/value head_dim {key.size(3)} != head_dim {self.head_dim}"
            )
        if key.device.type != self.device.type:
            raise ValueError(
                f"key device {key.device} != cache device {self.device}"
            )
        if key.dtype != self.dtype:
            raise ValueError(f"key dtype {key.dtype} != cache dtype {self.dtype}")

        batch, t_new = key.size(0), key.size(2)
        if t_new == 0:
            return batch, t_new
        if positions is not None:
            if not isinstance(positions, torch.Tensor):
                raise TypeError("positions must be a torch.Tensor")
            if positions.ndim != 1 or positions.numel() != t_new:
                raise ValueError(
                    f"positions must be 1-D with {t_new} entries, got "
                    f"{tuple(positions.shape)}"
                )
            if positions.dtype != torch.long:
                raise TypeError("positions must have dtype torch.long")
            if positions.numel() > 0 and bool((positions[1:] - positions[:-1] != 1).any()):
                raise ValueError("positions must be contiguous (no gaps or overlaps)")
        return batch, t_new

    @staticmethod
    def _tensor_bytes(tensors: list[torch.Tensor | None]) -> float:
        return float(
            sum(
                t.numel() * t.element_size()
                for t in tensors
                if t is not None
            )
        )

    @staticmethod
    def _dtype_size(dtype: torch.dtype) -> int:
        """Bytes occupied by a single element of ``dtype``."""
        if dtype.is_floating_point or dtype.is_complex:
            return torch.finfo(dtype).bits // 8
        return torch.iinfo(dtype).bits // 8

    @staticmethod
    def _resolve_positions(
        positions: torch.Tensor | None,
        t_new: int,
        start: torch.Tensor | int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the global positions ``[T_new]`` for an update call.

        If ``positions`` is given it is returned unchanged; otherwise the
        positions are derived from ``start`` (a per-row start, or a scalar
        start shared by every row).
        """
        if positions is not None:
            return positions
        if isinstance(start, torch.Tensor):
            start = int(start.min().item())
        return torch.arange(start, start + t_new, device=device, dtype=torch.long)
