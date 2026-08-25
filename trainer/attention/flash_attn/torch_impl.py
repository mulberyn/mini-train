import math
import torch


def flash_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    B, Hq, Nq, D = q.shape
    Hkv = k.shape[1]
    group_size = Hq // Hkv
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2))
    scores = scores * (1.0 / math.sqrt(D))
    if causal:
        q_pos = torch.arange(Nq, device=q.device)
        k_pos = torch.arange(k.shape[-2], device=k.device)
        mask = k_pos[None, :] > q_pos[:, None]
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)