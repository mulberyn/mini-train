import pytest
import torch
import torch.nn.functional as F
from trainer.loss.cross_entropy import cross_entropy


def reference_cross_entropy(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@pytest.mark.parametrize("shape", [(2, 5), (8, 100), (4, 16, 100), (2, 128, 1000)])
def test_cross_entropy(shape):
    vocab_size = shape[-1]
    logits = torch.randn(*shape, dtype=torch.float32)
    targets = torch.randint(0, vocab_size, shape[:-1], dtype=torch.long)
    actual = cross_entropy(logits, targets)
    expected = reference_cross_entropy(logits, targets)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_cross_entropy_large_logits():
    logits = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]])
    targets = torch.tensor([1, 2])
    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(logits, targets)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cross_entropy_small_logits():
    logits = torch.tensor([[-1000.0, -1001.0, -999.0], [-2000.0, -1999.0, -2001.0]])
    targets = torch.tensor([0, 1])
    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(logits, targets)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cross_entropy_gradient():
    logits = torch.randn(4, 10, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 10, (4,))
    actual = cross_entropy(logits, targets)
    actual.backward()
    grad = logits.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_cross_entropy_gradcheck():
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 2, 4], dtype=torch.long)
    assert torch.autograd.gradcheck(lambda x: cross_entropy(x, targets), (logits,), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cross_entropy_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    logits = torch.randn(8, 1000, device=device)
    targets = torch.randint(0, 1000, (8,), device=device)
    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(logits, targets)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cross_entropy_dtype(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    logits = torch.randn(8, 1024, device="cuda", dtype=dtype)
    targets = torch.randint(0, 1024, (8,), device="cuda")
    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(logits, targets)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)