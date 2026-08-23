import math

import torch
from torch import nn

from trainer.layers.softmax import Softmax


def causal_mask(
    seq_len_q: int,
    seq_len_k: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    q_positions = torch.arange(seq_len_q, device=device)
    k_positions = torch.arange(seq_len_k, device=device)
    return k_positions.unsqueeze(0) > q_positions.unsqueeze(1)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    attention = ScaledDotProductAttention(causal=causal)
    return attention(q, k, v)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.softmax = Softmax(dim=-1)
    

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-1, -2))
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = scores * scale
        if self.causal:
            mask = causal_mask(seq_len_q=q.shape[-2], seq_len_k=k.shape[-2], device=q.device)
            scores = scores.masked_fill(mask, float("-inf"))
        probs = self.softmax(scores)
        output = torch.matmul(probs, v)
        return output