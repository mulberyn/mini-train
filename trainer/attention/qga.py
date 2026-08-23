import torch
from torch import nn
from einops import rearrange
from trainer.layers.linear import Linear
from trainer.attention.attention import scaled_dot_product_attention


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_groups: int,
        positional_encoding: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.d_k = d_model // num_heads
        self.heads_per_group = num_heads // num_groups
        self.wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wk = Linear(d_model, num_groups * self.d_k, device=device, dtype=dtype)
        self.wv = Linear(d_model, num_groups * self.d_k, device=device, dtype=dtype)
        self.wo = Linear(d_model, d_model, device=device, dtype=dtype)
        self.pos_enc = positional_encoding


    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        q = self.wq(x)          # (..., seq, d_model)
        k = self.wk(x)          # (..., seq, num_groups * d_k)
        v = self.wv(x)          # (..., seq, num_groups * d_k)
        q = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads, d=self.d_k)
        k = rearrange(k, "... s (g d) -> ... g s d", g=self.num_groups, d=self.d_k)
        v = rearrange(v, "... s (g d) -> ... g s d", g=self.num_groups, d=self.d_k)
        k = k.repeat_interleave(self.heads_per_group, dim=-3)  # 在 group 维度上重复
        v = v.repeat_interleave(self.heads_per_group, dim=-3)
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