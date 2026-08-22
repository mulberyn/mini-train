import torch


def cross_entropy(
    out_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    assert out_logits.shape[:-1] == targets.shape
    log_probs = out_logits - torch.logsumexp(out_logits, dim=-1, keepdim=True)
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()