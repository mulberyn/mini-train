"""Inference-time transformer that reads/writes a KV cache.

``KVCachedTransformerLM`` mirrors ``trainer.model.transformer.TransformerLM``
structurally and uses identical parameter names, so a trained model's
``state_dict`` can be copied in with a plain ``load_state_dict``. The
difference is the forward pass: every block receives the shared KV cache and
appends its new K/V before attending, so prefill/decode only compute the
current tokens instead of re-running the whole sequence.
"""

from __future__ import annotations

import torch
from torch import nn

from inference.attention.kv_attention import KVAttention
from inference.kv_cache.base import KVCache
from trainer.layers import Embedding, RMSNorm, Linear, RoPE, SwiGLU


class KVTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Submodule names mirror trainer.model.transformer_block.TransformerBlock.
        self.norm1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        rope = RoPE(
            theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.mha = KVAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache,
        layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.mha(self.norm1(x), kv_cache, layer_idx, positions)
        x = x + self.ffn(self.norm2(x))
        return x


class KVCachedTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        num_kv_heads: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_kv_heads is not None and d_model % num_kv_heads != 0:
            raise ValueError("d_model must be divisible by num_kv_heads")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.d_ff = d_ff

        # Same parameter names as trainer.model.transformer.TransformerLM.
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList(
            [
                KVTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        inputs: torch.Tensor,
        kv_cache: KVCache,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with KV caching.

        Args:
            inputs: ``[B, S]`` token ids (prefill chunk or one decode token).
            kv_cache: cache to append to and attend over. It must be sized for
                ``num_layers`` layers and ``num_kv_heads`` heads.
            positions: absolute token positions ``[S]`` shared by all rows;
                defaults to ``arange(S)`` (fresh prefill from position 0).

        Returns:
            ``[B, S, vocab_size]`` logits.
        """
        if inputs.ndim != 2:
            raise ValueError(
                f"inputs must have shape [B, S], got {inputs.shape}"
            )
        if inputs.dtype != torch.long:
            raise TypeError(f"inputs must be torch.long, got {inputs.dtype}")
        if inputs.size(1) > self.context_length:
            raise ValueError(
                f"sequence length {inputs.size(1)} exceeds context length "
                f"{self.context_length}"
            )
        if kv_cache.num_layers != self.num_layers:
            raise ValueError(
                f"kv_cache has {kv_cache.num_layers} layers but the model has "
                f"{self.num_layers}"
            )
        if kv_cache.num_kv_heads != self.num_kv_heads:
            raise ValueError(
                f"kv_cache has {kv_cache.num_kv_heads} kv heads but the model "
                f"expects {self.num_kv_heads}"
            )

        if positions is None:
            positions = torch.arange(inputs.size(1), device=inputs.device, dtype=torch.long)

        x = self.token_embedding(inputs)
        for layer_idx, block in enumerate(self.transformer_blocks):
            x = block(x, kv_cache, layer_idx, positions)
        x = self.output_norm(x)
        logits = self.lm_head(x)
        return logits
