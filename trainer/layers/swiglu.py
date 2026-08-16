import torch
from torch import nn

from trainer.layers.linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def calculate_ffn_dim(d_model: int) -> int:
    d_ff = int(8 * d_model / 3)
    return ((d_ff + 63) // 64) * 64


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_ff is None:
            d_ff = calculate_ffn_dim(d_model)
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)
        