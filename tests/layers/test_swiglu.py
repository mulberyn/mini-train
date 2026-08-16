import pytest
import torch
import torch.nn.functional as F
from trainer.layers.swiglu import SwiGLU, silu, calculate_ffn_dim


def make_input(batch_size=2, seq_len=16, d_model=128, dtype=torch.float32, device="cpu"):
    return torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device)


def test_silu_matches_pytorch():
    torch.manual_seed(42)
    x = torch.randn(4, 16, dtype=torch.float32)
    output = silu(x)
    expected = F.silu(x)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_silu_dtype(dtype):
    x = torch.randn(4, 16, dtype=dtype)
    output = silu(x)
    assert output.dtype == dtype
    assert output.shape == x.shape


@pytest.mark.parametrize("d_model", [128, 256, 512, 768, 1024, 2048, 4096])
def test_calculate_ffn_dim_is_multiple_of_64(d_model):
    d_ff = calculate_ffn_dim(d_model)
    assert d_ff % 64 == 0
    assert d_ff > 0


def test_default_ffn_dimension():
    d_model = 4096
    d_ff = calculate_ffn_dim(d_model)
    assert d_ff == 10944


@pytest.mark.parametrize("batch_size,seq_len,d_model,d_ff", [(1, 1, 64, 128), (1, 8, 128, 256), (2, 16, 128, 256), (4, 32, 256, 512), (2, 64, 512, 1024)])
def test_swiglu_shape(batch_size, seq_len, d_model, d_ff):
    model = SwiGLU(d_model=d_model, d_ff=d_ff)
    x = make_input(batch_size=batch_size, seq_len=seq_len, d_model=d_model)
    output = model(x)
    assert output.shape == x.shape


def test_swiglu_supports_arbitrary_leading_dimensions():
    model = SwiGLU(d_model=128, d_ff=256)
    x = torch.randn(2, 4, 8, 128)
    output = model(x)
    assert output.shape == x.shape


def test_swiglu_matches_manual_computation():
    torch.manual_seed(42)
    d_model = 32
    d_ff = 64
    model = SwiGLU(d_model=d_model, d_ff=d_ff)
    x = torch.randn(2, 8, d_model)
    output = model(x)
    gate = F.silu(model.w1(x))
    up = model.w3(x)
    expected = model.w2(gate * up)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


def test_swiglu_backward():
    torch.manual_seed(42)
    model = SwiGLU(d_model=64, d_ff=128)
    x = torch.randn(2, 8, 64, requires_grad=True)
    output = model(x)
    loss = output.mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient contains NaN/Inf"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_swiglu_dtype(dtype):
    model = SwiGLU(d_model=64, d_ff=128, dtype=dtype)
    x = torch.randn(2, 8, 64, dtype=dtype)
    output = model(x)
    assert output.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_swiglu_cuda():
    device = torch.device("cuda")
    model = SwiGLU(d_model=128, d_ff=256, device=device)
    x = torch.randn(2, 16, 128, device=device)
    output = model(x)
    assert output.device.type == "cuda"
    assert output.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_swiglu_cuda_dtype(dtype):
    device = torch.device("cuda")
    model = SwiGLU(d_model=128, d_ff=256, device=device, dtype=dtype)
    x = torch.randn(2, 16, 128, device=device, dtype=dtype)
    output = model(x)
    assert output.device.type == "cuda"
    assert output.dtype == dtype
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 10.0, -10.0, 50.0, -50.0])
def test_silu_numerical_stability(value):
    x = torch.tensor([value], dtype=torch.float32)
    output = silu(x)
    assert torch.isfinite(output).all()


def test_swiglu_numerical_stability():
    model = SwiGLU(d_model=64, d_ff=128)
    x = torch.randn(2, 8, 64) * 20
    output = model(x)
    assert torch.isfinite(output).all()


def test_parameter_count():
    d_model = 128
    d_ff = 256
    model = SwiGLU(d_model=d_model, d_ff=d_ff)
    expected = 3 * d_model * d_ff
    actual = sum(p.numel() for p in model.parameters())
    assert actual == expected


def test_invalid_d_model():
    with pytest.raises(ValueError):
        SwiGLU(d_model=0)


def test_invalid_d_ff():
    with pytest.raises(ValueError):
        SwiGLU(d_model=128, d_ff=0)