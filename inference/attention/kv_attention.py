"""Inference-only multi-head attention that reads/writes a KV cache.

This is the inference counterpart of ``trainer.attention.mha.MultiHeadAttention``:
the parameters (``wq/wk/wv/wo``) and module names are identical so weights can
be copied straight from a trained model, but the forward pass appends the new
K/V to a :class:`~inference.kv_cache.base.KVCache` and then attends over the
*cached* prefix instead of recomputing over the whole sequence.

The module keeps the cache in sync: after ``forward`` the cache contains the
K/V of every token seen so far, so the next decode step can reuse them.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from einops import rearrange

from inference.attention.attention import kv_causal_mask
from inference.kv_cache.base import KVCache
from trainer.layers.linear import Linear


class KVAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoding: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Parameter names mirror trainer.attention.mha.MultiHeadAttention so that
        # load_state_dict(model.state_dict()) copies weights directly.
        self.wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wo = Linear(d_model, d_model, device=device, dtype=dtype)
        self.pos_enc = positional_encoding

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cached attention over the full key/value prefix.

        Args:
            x: ``[B, Sq, d_model]`` input (query tokens).
            kv_cache: the KV cache to read from and write to.
            layer_idx: which layer's K/V to use.
            positions: absolute token positions ``[Sq]`` of the input tokens
                (shared by every batch row); defaults to ``arange(Sq)``.

        Returns:
            ``[B, Sq, d_model]`` attention output.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input dimension {self.d_model}, but got {x.shape[-1]}")
        batch, seq_len = x.shape[0], x.shape[1]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)

        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            positions = positions.to(x.device)
            if positions.dtype != torch.long:
                raise TypeError("positions must have dtype torch.long")
            if positions.numel() != seq_len:
                raise ValueError(
                    f"positions has {positions.numel()} entries but input has "
                    f"{seq_len} tokens"
                )

        if self.pos_enc is not None:
            q = self.pos_enc(q, positions)
            k = self.pos_enc(k, positions)

        # Append this step's K/V to the cache, then attend over the whole
        # cached prefix (past tokens + this step's tokens).
        kv_cache.update(layer_idx, k, v, positions)
        k_cached, v_cached = kv_cache.get(layer_idx)

        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(q, k_cached.transpose(-1, -2)) * scale
        mask = kv_causal_mask(positions, k_cached.shape[-2], device=x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v_cached)

        attn = rearrange(attn, "... h s d -> ... s (h d)")
        return self.wo(attn)
