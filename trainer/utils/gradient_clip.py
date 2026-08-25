import torch
from collections.abc import Iterable


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0
) -> float: # L2 grad norm
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if norm_type <= 0:
        raise ValueError("norm_type must be positive")
    
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    
    if norm_type == float("inf"):
        total_norm = max(
            g.detach().abs().max().item()
            for g in grads
        )
    else:
        total_norm = 0.0
        for g in grads:
            grad_norm = torch.linalg.vector_norm(g.detach(), ord=norm_type)
            total_norm += grad_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        with torch.no_grad():
            for g in grads:
                g.mul_(scale)

    return total_norm