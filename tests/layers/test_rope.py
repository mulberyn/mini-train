import pytest
import torch
from trainer.layers.rope import RoPE


def reference_rope(x: torch.Tensor, token_positions: torch.Tensor, theta: float) -> torch.Tensor:
    d_k = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=x.device, dtype=torch.float32) / d_k))
    freqs = torch.outer(token_positions.float(), inv_freq)
    cos = freqs.cos().to(x.dtype)
    sin = freqs.sin().to(x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.empty_like(x)
    y[..., 0::2] = y1
    y[..., 1::2] = y2
    return y


def test_rope_shape():
    B, H, S, D = 2, 4, 16, 32
    x = torch.randn(B, H, S, D)
    positions = torch.arange(S)
    rope = RoPE(theta=10000.0, d_k=D, max_seq_len=S)
    y = rope(x, positions)
    assert y.shape == x.shape


def test_rope_position_zero():
    B, H, S, D = 2, 4, 16, 32
    x = torch.randn(B, H, S, D)
    positions = torch.zeros(S, dtype=torch.long)
    rope = RoPE(theta=10000.0, d_k=D, max_seq_len=S)
    y = rope(x, positions)
    torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-6)


def test_rope_matches_reference():
    torch.manual_seed(42)
    B, H, S, D = 2, 4, 16, 32
    theta = 10000.0
    x = torch.randn(B, H, S, D)
    positions = torch.arange(S)
    rope = RoPE(theta=theta, d_k=D, max_seq_len=S)
    y = rope(x, positions)
    y_ref = reference_rope(x, positions, theta)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)


def test_rope_norm_preservation():
    B, H, S, D = 2, 4, 16, 32
    x = torch.randn(B, H, S, D)
    positions = torch.arange(S)
    rope = RoPE(theta=10000.0, d_k=D, max_seq_len=S)
    y = rope(x, positions)
    x_norm = torch.linalg.vector_norm(x, dim=-1)
    y_norm = torch.linalg.vector_norm(y, dim=-1)
    torch.testing.assert_close(x_norm, y_norm, rtol=1e-5, atol=1e-6)


def test_rope_different_positions():
    B, H, S, D = 1, 1, 8, 32
    x = torch.randn(B, H, S, D)
    positions = torch.arange(S)
    rope = RoPE(theta=10000.0, d_k=D, max_seq_len=S)
    y = rope(x, positions)
    assert not torch.allclose(y[:, :, 0], y[:, :, 1])


def test_rope_custom_positions():
    B, H, S, D = 2, 4, 4, 32
    x = torch.randn(B, H, S, D)
    positions = torch.tensor([0, 3, 7, 15], dtype=torch.long)
    rope = RoPE(theta=10000.0, d_k=D, max_seq_len=32)
    y = rope(x, positions)
    y_ref = reference_rope(x, positions, 10000.0)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rope_dtype(dtype):
    B, H, S, D = 2, 4, 16, 32
    x = torch.randn(B, H, S, D, dtype=dtype)
    positions = torch.arange(S)
    rope = RoPE(theta=10000.0, d_k=D, max_seq_len=S)
    y = rope(x, positions)
    assert y.dtype == dtype


def test_rope_requires_even_dim():
    with pytest.raises(AssertionError):
        RoPE(theta=10000.0, d_k=31, max_seq_len=128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rope_cuda():
    device = torch.device("cuda")
    B, H, S, D = 2, 4, 16, 32
    x = torch.randn(B, H, S, D, device=device)
    positions = torch.arange(S, device=device)
    rope = RoPE(theta=10000.0, d_k=D, max_seq_len=S, device=device)
    y = rope(x, positions)
    assert y.device.type == "cuda"
    assert y.shape == x.shape