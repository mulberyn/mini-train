import pytest
import torch
from torch import nn
from trainer.layers.rmsnorm import RMSNorm

def test_rmsnorm_shape():
    batch_size = 4
    seq_len = 16
    dim = 128
    x = torch.randn(batch_size, seq_len, dim)
    layer = RMSNorm(dim)
    y = layer(x)
    assert y.shape == x.shape


def test_rmsnorm_forward_matches_pytorch():
    torch.manual_seed(42)
    dim = 128
    eps = 1e-5
    x = torch.randn(4, 16, dim, dtype=torch.float32)
    reference = nn.RMSNorm(dim, eps=eps)
    layer = RMSNorm(dim, eps=eps)
    layer.weight.data.copy_(reference.weight.data)
    y = layer(x)
    y_ref = reference(x)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)


def test_rmsnorm_backward():
    torch.manual_seed(42)
    dim = 64
    eps = 1e-5
    x = torch.randn(4, 16, dim, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone()
    x_ref.requires_grad_(True)
    reference = nn.RMSNorm(dim, eps=eps)
    layer = RMSNorm(dim, eps=eps)
    layer.weight.data.copy_(reference.weight.data)
    y = layer(x)
    y_ref = reference(x_ref)
    grad_output = torch.randn_like(y)
    y.backward(grad_output)
    y_ref.backward(grad_output)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(layer.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-6)


def test_rmsnorm_zero_input():
    dim = 64
    x = torch.zeros(4, 16, dim)
    layer = RMSNorm(dim)
    y = layer(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.zeros_like(y))


def test_rmsnorm_small_input():
    dim = 64
    x = torch.full((4, 16, dim), 1e-10)
    layer = RMSNorm(dim)
    y = layer(x)
    assert torch.isfinite(y).all()


def test_rmsnorm_large_input():
    dim = 64
    x = torch.full((4, 16, dim), 1e4)
    layer = RMSNorm(dim)
    y = layer(x)
    assert torch.isfinite(y).all()


def test_rmsnorm_eps():
    dim = 64
    x = torch.randn(4, 16, dim)
    for eps in [1e-5, 1e-6, 1e-8]:
        layer = RMSNorm(dim, eps=eps)
        y = layer(x)
        assert torch.isfinite(y).all()


@pytest.mark.parametrize("shape", [(32,), (4, 32), (4, 16, 32), (2, 4, 16, 32)])
def test_rmsnorm_different_shapes(shape):
    dim = shape[-1]
    x = torch.randn(*shape)
    layer = RMSNorm(dim)
    y = layer(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rmsnorm_dtype(dtype):
    dim = 64
    layer = RMSNorm(dim, dtype=dtype)
    x = torch.randn(4, 16, dim, dtype=dtype)
    y = layer(x)
    assert y.dtype == dtype


def test_rmsnorm_weight_parameter():
    dim = 128
    layer = RMSNorm(dim)
    assert isinstance(layer.weight, nn.Parameter)
    assert layer.weight.requires_grad
    assert layer.weight.shape == (dim,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rmsnorm_cuda():
    dim = 128
    device = torch.device("cuda")
    layer = RMSNorm(dim, device=device)
    x = torch.randn(4, 16, dim, device=device)
    y = layer(x)
    assert y.device.type == "cuda"
    assert y.shape == x.shape
    assert torch.isfinite(y).all()