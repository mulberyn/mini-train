import math

import torch
from torch import nn

from trainer.layers.softmax import Softmax
from trainer.attention.mask import causal_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.softmax = Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.ndim != 4:
            raise ValueError(f"q must have shape [B, H, Sq, D], but got {q.shape}")
        if k.ndim != 4:
            raise ValueError(f"k must have shape [B, H, Sk, D], but got {k.shape}")
        if v.ndim != 4:
            raise ValueError(f"v must have shape [B, H, Sk, D], but got {v.shape}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("Batch size of q, k, v must be the same.")
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            raise ValueError("Number of heads of q, k, v must be the same.")
        if k.shape[-2] != v.shape[-2]:
            raise ValueError("Sequence length of k and v must be the same.")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("Head dimension of q and k must be the same.")

        scores = torch.matmul(q, k.transpose(-1, -2))
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = scores * scale

        if self.causal:
            mask = causal_mask(seq_len_q=q.shape[-2], seq_len_k=k.shape[-2], device=q.device)
            scores = scores.masked_fill(mask, float("-inf"))

        probs = self.softmax(scores)
        output = torch.matmul(probs, v)
        return output


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    attention = ScaledDotProductAttention(causal=causal)
    return attention(q, k, v)