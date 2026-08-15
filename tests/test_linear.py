import pytest
import torch
from torch import nn
from trainer.layers.linear import Linear

def test_linear_shape():
    batch_size = 4
    seq_len = 16
    in_features = 32
    out_features = 64
    x = torch.randn(batch_size, seq_len, in_features)
    layer = Linear(in_features, out_features)
    y = layer(x)
    assert y.shape == (batch_size, seq_len, out_features)


def test_linear_matches_pytorch():
    torch.manual_seed(42)
    in_features = 32
    out_features = 64
    x = torch.randn(8, in_features)
    reference = nn.Linear(in_features, out_features, bias=False)
    layer = Linear(in_features, out_features)
    layer.weight.data.copy_(reference.weight.data)
    y = layer(x)
    y_ref = reference(x)
    torch.testing.assert_close(y, y_ref)


def test_linear_backward():
    torch.manual_seed(42)
    batch_size = 8
    seq_len = 16
    in_features = 32
    out_features = 64
    x = torch.randn(batch_size, seq_len, in_features, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    reference = nn.Linear(in_features, out_features, bias=False)
    layer = Linear(in_features, out_features)
    layer.weight.data.copy_(reference.weight.data)
    y = layer(x)
    y_ref = reference(x_ref)
    grad_output = torch.randn_like(y)
    y.backward(grad_output)
    y_ref.backward(grad_output)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(layer.weight.grad, reference.weight.grad)


def test_linear_dtype():
    for dtype in [torch.float32, torch.float64]:
        layer = Linear(in_features=32, out_features=64, dtype=dtype)
        x = torch.randn(8, 32, dtype=dtype)
        y = layer(x)
        assert y.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_linear_cuda():
    device = torch.device("cuda")
    layer = Linear(in_features=32, out_features=64, device=device)
    x = torch.randn(8, 32, device=device)
    y = layer(x)
    assert y.device.type == "cuda"
    assert y.shape == (8, 64)


def test_linear_parameter():
    layer = Linear(in_features=32, out_features=64)
    assert isinstance(layer.weight, nn.Parameter)
    assert layer.weight.requires_grad
    assert layer.weight.shape == (64, 32)