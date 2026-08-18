import pytest
import torch
from trainer.model.transformer_block import TransformerBlock


def make_block(d_model=128, num_heads=4, d_ff=256, max_seq_len=128, theta=10000.0, device="cpu", dtype=torch.float32):
    return TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)


def make_input(batch_size=2, seq_len=16, d_model=128, device="cpu", dtype=torch.float32, requires_grad=False):
    return torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad)


@pytest.mark.parametrize("batch_size,seq_len,d_model,num_heads,d_ff", [(1, 1, 64, 4, 128), (1, 8, 128, 4, 256), (2, 16, 128, 4, 256), (4, 32, 256, 8, 512), (2, 64, 512, 8, 1024)])
def test_transformer_block_shape(batch_size, seq_len, d_model, num_heads, d_ff):
    block = make_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=seq_len)
    x = make_input(batch_size=batch_size, seq_len=seq_len, d_model=d_model)
    output = block(x)
    assert output.shape == x.shape


def test_transformer_block_token_positions():
    torch.manual_seed(42)
    block = make_block(d_model=128, num_heads=4, d_ff=256, max_seq_len=128)
    x = make_input(batch_size=2, seq_len=16, d_model=128)
    token_positions = torch.arange(16, device=x.device)
    output = block(x, token_positions=token_positions)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_default_token_positions_match_explicit_positions():
    torch.manual_seed(42)
    block = make_block(d_model=64, num_heads=4, d_ff=128, max_seq_len=128)
    x = make_input(batch_size=2, seq_len=8, d_model=64)
    output_default = block(x)
    positions = torch.arange(8, device=x.device)
    output_explicit = block(x, token_positions=positions)
    torch.testing.assert_close(output_default, output_explicit, rtol=1e-5, atol=1e-6)


def test_transformer_block_output_is_finite():
    block = make_block()
    x = make_input()
    output = block(x)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 10.0, -10.0])
def test_transformer_block_numerical_stability(value):
    block = make_block()
    x = torch.full((2, 16, 128), value)
    output = block(x)
    assert torch.isfinite(output).all()


def test_residual_connection():
    block = make_block(d_model=64, num_heads=4, d_ff=128, max_seq_len=32)
    x = make_input(batch_size=2, seq_len=8, d_model=64)

    class ZeroModule(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return torch.zeros_like(x)

    block.mha = ZeroModule()
    block.ffn = ZeroModule()
    output = block(x)
    torch.testing.assert_close(output, x)


def test_transformer_block_backward():
    torch.manual_seed(42)
    block = make_block(d_model=64, num_heads=4, d_ff=128, max_seq_len=32)
    x = make_input(batch_size=2, seq_len=8, d_model=64, requires_grad=True)
    output = block(x)
    loss = output.mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_all_parameters_receive_gradients():
    torch.manual_seed(42)
    block = make_block(d_model=64, num_heads=4, d_ff=128, max_seq_len=32)
    x = make_input(batch_size=2, seq_len=8, d_model=64)
    output = block(x)
    loss = output.square().mean()
    loss.backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"{name} did not receive gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient contains NaN/Inf"


def test_gradient_is_not_zero():
    torch.manual_seed(42)
    block = make_block(d_model=64, num_heads=4, d_ff=128, max_seq_len=32)
    x = make_input(batch_size=2, seq_len=8, d_model=64, requires_grad=True)
    output = block(x)
    loss = output.square().mean()
    loss.backward()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transformer_block_dtype(dtype):
    block = make_block(d_model=64, num_heads=4, d_ff=128, max_seq_len=32, dtype=dtype)
    x = make_input(batch_size=2, seq_len=8, d_model=64, dtype=dtype)
    output = block(x)
    assert output.dtype == dtype
    assert output.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_transformer_block_cuda():
    device = torch.device("cuda")
    block = make_block(d_model=128, num_heads=4, d_ff=256, max_seq_len=64, device=device, dtype=torch.float32)
    x = make_input(batch_size=2, seq_len=16, d_model=128, device=device, dtype=torch.float32)
    output = block(x)
    assert output.device.type == "cuda"
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_transformer_block_cuda_dtype(dtype):
    device = torch.device("cuda")
    block = make_block(d_model=128, num_heads=4, d_ff=256, max_seq_len=64, device=device, dtype=dtype)
    x = make_input(batch_size=2, seq_len=16, d_model=128, device=device, dtype=dtype)
    output = block(x)
    assert output.device.type == "cuda"
    assert output.dtype == dtype
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_transformer_block_cuda_backward():
    device = torch.device("cuda")
    block = make_block(d_model=128, num_heads=4, d_ff=256, max_seq_len=64, device=device, dtype=torch.bfloat16)
    x = make_input(batch_size=2, seq_len=16, d_model=128, device=device, dtype=torch.bfloat16, requires_grad=True)
    output = block(x)
    loss = output.float().square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"{name} gradient is None"
        assert torch.isfinite(param.grad).all(), f"{name} gradient contains NaN/Inf"


def test_transformer_block_parameter_count():
    d_model = 128
    d_ff = 256
    num_heads = 4
    block = make_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=64)
    actual = sum(p.numel() for p in block.parameters())
    expected = 2 * d_model + 4 * d_model * d_model + 3 * d_model * d_ff
    assert actual == expected


def test_invalid_d_model():
    with pytest.raises(ValueError):
        make_block(d_model=0, num_heads=4, d_ff=128)


def test_invalid_num_heads():
    with pytest.raises(ValueError):
        make_block(d_model=128, num_heads=0, d_ff=256)


def test_d_model_not_divisible_by_num_heads():
    with pytest.raises(ValueError):
        make_block(d_model=130, num_heads=4, d_ff=256)


def test_invalid_d_ff():
    with pytest.raises(ValueError):
        make_block(d_model=128, num_heads=4, d_ff=0)


def test_invalid_max_seq_len():
    with pytest.raises(ValueError):
        make_block(d_model=128, num_heads=4, d_ff=256, max_seq_len=0)