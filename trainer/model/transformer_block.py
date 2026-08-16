import torch
from torch import nn

from trainer.layers import RMSNorm, SwiGLU, RoPE
from trainer.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
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
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        self.norm1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        rope = RoPE(
            theta, 
            d_k=d_model // num_heads, 
            max_seq_len=max_seq_len, 
            device=device, 
            dtype=dtype
        )
        self.mha = MultiHeadAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    
    def forward(
        self, 
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.mha(self.norm1(x), token_positions)
        x = x + self.ffn(self.norm2(x))
        return x