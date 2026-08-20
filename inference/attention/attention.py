"""Reference attention helpers for inference.

The inference engine keeps its own attention code separate from the training
path (``trainer.attention``) because cached decoding needs a different masking
scheme: the query tokens and the cached key tokens have *absolute* positions
that are no longer aligned, so the mask must be built from positions instead
of a fixed lower-triangular matrix.
"""

from __future__ import annotations

import math

import torch


def kv_causal_mask(
    query_positions: torch.Tensor,
    kv_length: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Causal mask ``[B, Sq, kv_length]`` for cached attention.

    A query token at global position ``p`` may attend to cached key position
    ``j`` only if ``j <= p``. Returns a boolean tensor where ``True`` means
    "mask out" (i.e. key position ``j > p``).

    Args:
        query_positions: ``[B, Sq]`` (or ``[Sq]``) absolute positions of the
            query tokens, shared by every batch row.
        kv_length: number of cached key positions ``0 .. kv_length-1``.
    """
    if query_positions.ndim == 1:
        query_positions = query_positions.unsqueeze(0)
    if device is None:
        device = query_positions.device
    kv_positions = torch.arange(kv_length, device=device, dtype=torch.long)
    return kv_positions[None, None, :] > query_positions[..., None]  # [B, Sq, kv]


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Standard softmax attention over dense K/V.

    Args:
        query: ``[B, H, Sq, D]``
        key: ``[B, H, Sk, D]``
        value: ``[B, H, Sk, D]``
        mask: optional boolean ``[B, Sq, Sk]`` (True = mask out).
        scale: score scale; defaults to ``1 / sqrt(D)``.

    Returns:
        ``[B, H, Sq, D]``
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"q/k/v must be 4-D [B, H, S, D], got "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
        )
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dims must match")

    scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value)


def attention_with_positions(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions: torch.Tensor | None,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal attention where keys/values may contain *future* positions.

    Unlike plain causal attention (which assumes key positions ``0..Sk-1``),
    the key tensor may be a full cached prefix that extends past the query
    tokens, so the mask is derived from the absolute positions of both sides.

    Args:
        query: ``[B, H, Sq, D]``
        key/value: ``[B, H, Sk, D]``
        query_positions: ``[B, Sq]`` or ``[Sq]`` absolute query positions.
    """
    if query_positions is None:
        query_positions = torch.arange(query.shape[-2], device=query.device)
    mask = kv_causal_mask(query_positions, key.shape[-2], device=query.device)
    return dense_attention(query, key, value, mask=mask, scale=scale)
