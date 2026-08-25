import torch

from .torch_impl import flash_attention_torch
from .triton_impl import flash_attention_triton


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    backend: str = "auto",
) -> torch.Tensor:
    if backend == "auto":
        if q.is_cuda:
            try:
                from .cuda import cuda_available
                if cuda_available():
                    backend = "cuda"
                elif q.is_cuda:
                    backend = "triton"
                else:
                    backend = "torch"
            except ImportError:
                backend = "triton"
        else:
            backend = "torch"

    if backend == "torch":
        return flash_attention_torch(q, k, v, causal=causal)
    if backend == "triton":
        return flash_attention_triton(q, k, v, causal=causal)
    if backend == "cuda":
        from .cuda import flash_attention_cuda
        return flash_attention_cuda(q, k, v, causal=causal)
    raise ValueError(
        f"Unknown FlashAttention backend: {backend}"
    )