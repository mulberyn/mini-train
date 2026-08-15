import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(
        self, 
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even."
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype).float() / d_k))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        positions = torch.arange(max_seq_len, device=device)
        
        freqs = torch.einsum('i,j -> ij', positions.to(dtype), inv_freq)
        
        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)

        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None
    ) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)  # (..., d_k//2, 2)
        return rotated.view(*x.shape)