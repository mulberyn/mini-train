import pytest
import torch
from trainer.layers.softmax import Softmax


def test_softmax_sum_to_one():
    x = torch.randn(2, 4, 8)
    softmax = Softmax(dim=-1)
    y = softmax(x)
    sums = y.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)


def test_softmax_non_negative():
    x = torch.randn(2, 4, 8)
    softmax = Softmax(dim=-1)
    y = softmax(x)
    assert torch.all(y >= 0)


def test_softmax_matches_torch():
    torch.manual_seed(42)
    x = torch.randn(4, 8, 32)
    softmax = Softmax(dim=-1)
    y = softmax(x)
    expected = torch.softmax(x, dim=-1)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)


def test_softmax_numerical_stability():
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]])
    softmax = Softmax(dim=-1)
    y = softmax(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.sum(dim=-1), torch.ones(2), rtol=1e-5, atol=1e-6)


def test_softmax_large_range():
    x = torch.tensor([[-10000.0, 0.0, 10000.0]])
    softmax = Softmax(dim=-1)
    y = softmax(x)
    assert torch.isfinite(y).all()
    assert y[0, -1] > 0.99


def test_softmax_different_dim():
    x = torch.randn(2, 4, 8)
    softmax = Softmax(dim=1)
    y = softmax(x)
    sums = y.sum(dim=1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)


def test_softmax_shift_invariance():
    x = torch.randn(4, 32)
    softmax = Softmax(dim=-1)
    y1 = softmax(x)
    y2 = softmax(x + 100.0)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_softmax_dtype(dtype):
    x = torch.randn(4, 8, 32, dtype=dtype)
    softmax = Softmax(dim=-1)
    y = softmax(x)
    assert y.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_softmax_cuda():
    device = torch.device("cuda")
    x = torch.randn(4, 8, 1024, device=device)
    softmax = Softmax(dim=-1)
    y = softmax(x)
    expected = torch.softmax(x, dim=-1)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)