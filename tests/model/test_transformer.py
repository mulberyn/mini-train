import pytest
import torch
from trainer.model.transformer import TransformerLM


def make_model(
    vocab_size=1000,
    context_length=128,
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=128,
    device="cpu",
    dtype=None,
):
    return TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0,
        device=torch.device(device),
        dtype=dtype,
    )


def make_inputs(batch_size=2, seq_len=16, vocab_size=1000, device="cpu"):
    return torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)


def test_transformer_lm_output_shape():
    model = make_model(vocab_size=1000, context_length=128, d_model=64, num_layers=2, num_heads=4, d_ff=128)
    inputs = make_inputs(batch_size=2, seq_len=16, vocab_size=1000)
    logits = model(inputs)
    assert logits.shape == (2, 16, 1000)


@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (1, 8), (2, 16), (4, 32)])
def test_transformer_lm_shapes(batch_size, seq_len):
    vocab_size = 500
    model = make_model(vocab_size=vocab_size, context_length=64)
    inputs = make_inputs(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size)
    logits = model(inputs)
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_transformer_lm_supports_max_context_length():
    context_length = 32
    model = make_model(vocab_size=500, context_length=context_length)
    inputs = make_inputs(batch_size=2, seq_len=context_length, vocab_size=500)
    logits = model(inputs)
    assert logits.shape == (2, context_length, 500)


def test_transformer_lm_rejects_context_overflow():
    context_length = 32
    model = make_model(vocab_size=500, context_length=context_length)
    inputs = make_inputs(batch_size=2, seq_len=context_length + 1, vocab_size=500)
    with pytest.raises(ValueError):
        model(inputs)


def test_transformer_lm_rejects_wrong_input_rank():
    model = make_model()
    inputs = torch.randint(0, 1000, (2, 8, 1), dtype=torch.long)
    with pytest.raises(ValueError):
        model(inputs)


def test_transformer_lm_rejects_non_long_input():
    model = make_model()
    inputs = torch.randn(2, 8)
    with pytest.raises(TypeError):
        model(inputs)


def test_transformer_lm_rejects_invalid_token_ids():
    model = make_model(vocab_size=100)
    inputs = torch.tensor([[0, 1, 2, 100]], dtype=torch.long)
    with pytest.raises(IndexError):
        model(inputs)


def test_default_positions_match_explicit_positions():
    torch.manual_seed(42)
    model = make_model(vocab_size=500, context_length=128)
    inputs = make_inputs(batch_size=2, seq_len=16, vocab_size=500)
    output1 = model(inputs)
    positions = torch.arange(16, dtype=torch.long)
    output2 = model(inputs, token_positions=positions)
    torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)


def test_transformer_lm_backward():
    model = make_model(vocab_size=500, context_length=64)
    inputs = make_inputs(batch_size=2, seq_len=16, vocab_size=500)
    logits = model(inputs)
    loss = logits.mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient contains NaN or Inf"


def test_transformer_lm_has_parameters():
    model = make_model()
    num_parameters = sum(p.numel() for p in model.parameters())
    assert num_parameters > 0


def test_transformer_lm_deterministic():
    torch.manual_seed(42)
    model1 = make_model()
    inputs = make_inputs()
    output1 = model1(inputs)
    torch.manual_seed(42)
    model2 = make_model()
    output2 = model2(inputs)
    torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transformer_lm_dtype(dtype):
    model = make_model(dtype=dtype)
    inputs = make_inputs()
    logits = model(inputs)
    assert logits.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_transformer_lm_cuda():
    device = torch.device("cuda")
    model = make_model(vocab_size=1000, context_length=128, device=device, dtype=torch.float32)
    inputs = make_inputs(batch_size=2, seq_len=16, vocab_size=1000, device=device)
    logits = model(inputs)
    assert logits.device.type == "cuda"
    assert logits.shape == (2, 16, 1000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_transformer_lm_cuda_dtype(dtype):
    device = torch.device("cuda")
    model = make_model(vocab_size=1000, context_length=128, device=device, dtype=dtype)
    inputs = make_inputs(batch_size=2, seq_len=16, vocab_size=1000, device=device)
    logits = model(inputs)
    assert logits.device.type == "cuda"
    assert logits.dtype == dtype