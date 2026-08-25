import math

import torch
from torch import nn

from trainer.layers import Softmax, Linear
from einops import rearrange


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        positional_encoding: nn.Module | None = None,
        attention_backend: str = "torch",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = d_model // num_heads
        self.heads_per_group = num_heads // num_kv_heads
        self.attention_backend = attention_backend
        self.wq = Linear(d_model, num_heads * self.d_head, device=device, dtype=dtype)
        self.wk = Linear(d_model, num_kv_heads * self.d_head, device=device, dtype=dtype)
        self.wv = Linear(d_model, num_kv_heads * self.d_head, device=device, dtype=dtype)
        self.wo = Linear(d_model, d_model, device=device, dtype=dtype)
        self.pos_enc = positional_encoding
    
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        token_positions: torch.Tensor | None,
        x_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos_enc is None:
            return q, k
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x_device, dtype=torch.long)
        q = self.pos_enc(q, token_positions)
        k = self.pos_enc(k, token_positions)
        return q, k
        
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        q = self.wq(x)          # (..., seq, d_model)
        k = self.wk(x)          # (..., seq, num_kv_heads * d_k)
        v = self.wv(x)          # (..., seq, num_kv_heads * d_k)
        q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads, d=self.d_k)
        k = rearrange(k, "... s (g d) -> ... g s d", g=self.num_kv_heads, d=self.d_k)
        v = rearrange(v, "... s (g d) -> ... g s d", g=self.num_kv_heads, d=self.d_k)
        q, k = self._apply_rope(q, k, seq_len, token_positions, x.device)
        from trainer.attention.flash_attn import flash_attn
        attn = flash_attn(q, k, v, casual=True, backend=self.attention_backend)
        attn = rearrange(attn, "... h s d -> ... s (h d)")
        return self.wo(attn)