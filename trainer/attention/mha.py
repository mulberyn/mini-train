import torch
from torch import nn
from einops import rearrange
from trainer.layers.linear import Linear
from trainer.attention.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoding: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wo = Linear(d_model, d_model, device=device, dtype=dtype)
        self.pos_enc = positional_encoding


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)
        if self.pos_enc is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            else:
                if token_positions.device != x.device:
                    token_positions = token_positions.to(x.device)
                if token_positions.dtype != torch.long:
                    raise TypeError("token_positions must have dtype torch.long")
            q = self.pos_enc(q, token_positions)
            k = self.pos_enc(k, token_positions)
        attn = scaled_dot_product_attention(q, k, v, causal=True)
        attn = rearrange(attn, "... h s d -> ... s (h d)")
        out = self.wo(attn)
        return out