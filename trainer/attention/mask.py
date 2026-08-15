import torch


def causal_mask(
    seq_len_q: int,
    seq_len_k: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    q_positions = torch.arange(seq_len_q, device=device)
    k_positions = torch.arange(seq_len_k, device=device)
    return k_positions.unsqueeze(0) > q_positions.unsqueeze(1)