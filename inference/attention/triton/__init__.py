"""Triton kernels for inference attention."""

from inference.attention.triton.paged_attention import (
    TRITON_AVAILABLE,
    paged_attention_triton,
)

__all__ = ["TRITON_AVAILABLE", "paged_attention_triton"]
