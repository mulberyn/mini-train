import pytest
import torch
from trainer.attention.mha import MultiHeadAttention
from trainer.layers.rope import RoPE


def make_mha(d_model=128, num_heads=4, device="cpu", dtype=torch.float32, use_rope=False, max_seq_len=1024):
    positional_encoding = None
    if use_rope:
        positional_encoding = RoPE(theta=10000.0, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device, dtype=dtype)
    return MultiHeadAttention(d_model=d_model, num_heads=num_heads, positional_encoding=positional_encoding, device=device, dtype=dtype)


def make_input(batch_size, seq_len, d_model, device="cpu", dtype=torch.float32):
    return torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)


def test_mha_shape():
    mha = make_mha(d_model=128, num_heads=4)
    x = make_input(batch_size=2, seq_len=32, d_model=128)
    output = mha(x)
    assert output.shape == x.shape


@pytest.mark.parametrize("batch_size,seq_len,d_model,num_heads", [(1, 1, 64, 4), (1, 16, 128, 4), (2, 32, 128, 8), (4, 64, 256, 8), (2, 128, 512, 16)])
def test_mha_shapes(batch_size, seq_len, d_model, num_heads):
    mha = make_mha(d_model=d_model, num_heads=num_heads)
    x = make_input(batch_size, seq_len, d_model)
    output = mha(x)
    assert output.shape == (batch_size, seq_len, d_model)


@pytest.mark.parametrize("seq_len", [1, 2, 8, 32, 128, 256])
def test_mha_different_sequence_lengths(seq_len):
    mha = make_mha(d_model=128, num_heads=8)
    x = make_input(batch_size=2, seq_len=seq_len, d_model=128)
    output = mha(x)
    assert output.shape == x.shape


def test_mha_rejects_invalid_d_model():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=127, num_heads=8)


def test_mha_rejects_zero_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=128, num_heads=0)


def test_mha_rejects_invalid_input_dimension():
    mha = make_mha(d_model=128, num_heads=8)
    x = torch.randn(2, 32, 64)
    with pytest.raises(ValueError):
        mha(x)


def test_mha_is_causal():
    torch.manual_seed(42)
    mha = make_mha(d_model=64, num_heads=4)
    x = torch.randn(1, 8, 64)
    out1 = mha(x)
    x2 = x.clone()
    x2[:, 4:] = torch.randn_like(x2[:, 4:])
    out2 = mha(x2)
    torch.testing.assert_close(out1[:, :4], out2[:, :4], rtol=1e-5, atol=1e-6)


def test_mha_with_rope():
    mha = make_mha(d_model=128, num_heads=8, use_rope=True, max_seq_len=128)
    x = make_input(batch_size=2, seq_len=32, d_model=128)
    output = mha(x)
    assert output.shape == x.shape


def test_mha_without_rope():
    mha = make_mha(d_model=128, num_heads=8, use_rope=False)
    x = make_input(batch_size=2, seq_len=32, d_model=128)
    output = mha(x)
    assert output.shape == x.shape


def test_rope_changes_output():
    torch.manual_seed(42)
    mha_no_rope = make_mha(d_model=64, num_heads=4, use_rope=False)
    torch.manual_seed(42)
    mha_rope = make_mha(d_model=64, num_heads=4, use_rope=True, max_seq_len=128)
    mha_rope.load_state_dict(mha_no_rope.state_dict())
    x = torch.randn(1, 16, 64)
    out_no_rope = mha_no_rope(x)
    out_rope = mha_rope(x)
    assert not torch.allclose(out_no_rope, out_rope)


def test_mha_token_positions():
    mha = make_mha(d_model=64, num_heads=4, use_rope=True, max_seq_len=256)
    x = torch.randn(1, 8, 64)
    token_positions = torch.arange(10, 18, dtype=torch.long)
    output = mha(x, token_positions=token_positions)
    assert output.shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float32])
def test_mha_dtype(dtype):
    mha = make_mha(d_model=128, num_heads=8, dtype=dtype)
    x = make_input(batch_size=2, seq_len=32, d_model=128, dtype=dtype)
    output = mha(x)
    assert output.dtype == dtype


def test_mha_backward():
    mha = make_mha(d_model=128, num_heads=8)
    x = torch.randn(2, 32, 128, requires_grad=True)
    output = mha(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in mha.named_parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mha_cuda():
    device = torch.device("cuda")
    mha = make_mha(d_model=128, num_heads=8, device=device, dtype=torch.float32)
    x = make_input(batch_size=2, seq_len=64, d_model=128, device=device, dtype=torch.float32)
    output = mha(x)
    assert output.device == x.device
    assert output.dtype == x.dtype
    assert output.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mha_cuda_dtype(dtype):
    device = torch.device("cuda")
    mha = make_mha(d_model=128, num_heads=8, device=device, dtype=dtype)
    x = make_input(batch_size=2, seq_len=32, d_model=128, device=device, dtype=dtype)
    output = mha(x)
    assert output.device == x.device
    assert output.dtype == dtype
    assert output.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mha_cuda_backward():
    device = torch.device("cuda")
    mha = make_mha(d_model=128, num_heads=8, device=device, dtype=torch.float32)
    x = torch.randn(2, 32, 128, device=device, requires_grad=True)
    output = mha(x)
    loss = output.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in mha.named_parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()