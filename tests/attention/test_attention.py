import pytest
import torch
import torch.nn.functional as F
from trainer.attention.attention import scaled_dot_product_attention


def make_qkv(
    batch_size: int = 2,
    num_heads: int = 4,
    seq_len_q: int = 8,
    seq_len_k: int = 8,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
):
    q = torch.randn(batch_size, num_heads, seq_len_q, head_dim, dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn(batch_size, num_heads, seq_len_k, head_dim, dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn(batch_size, num_heads, seq_len_k, head_dim, dtype=dtype, device=device, requires_grad=requires_grad)
    return q, k, v


def test_attention_shape():
    B = 2
    H = 4
    Sq = 8
    Sk = 8
    D = 16
    q, k, v = make_qkv(batch_size=B, num_heads=H, seq_len_q=Sq, seq_len_k=Sk, head_dim=D)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    assert output.shape == (B, H, Sq, D)


def test_attention_shape_different_sequence_lengths():
    B = 2
    H = 4
    Sq = 5
    Sk = 8
    D = 16
    q, k, v = make_qkv(batch_size=B, num_heads=H, seq_len_q=Sq, seq_len_k=Sk, head_dim=D)
    output = scaled_dot_product_attention(q, k, v, causal=False)
    assert output.shape == (B, H, Sq, D)


@pytest.mark.parametrize("B,H,S,D", [(1, 1, 1, 4), (1, 1, 8, 16), (2, 4, 32, 32), (4, 8, 64, 64), (2, 16, 128, 128)])
def test_attention_shapes(B, H, S, D):
    q, k, v = make_qkv(batch_size=B, num_heads=H, seq_len_q=S, seq_len_k=S, head_dim=D)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    assert output.shape == (B, H, S, D)


def test_attention_matches_torch_causal():
    torch.manual_seed(42)
    q, k, v = make_qkv(batch_size=2, num_heads=4, seq_len_q=16, seq_len_k=16, head_dim=32)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_attention_matches_torch_non_causal():
    torch.manual_seed(42)
    q, k, v = make_qkv(batch_size=2, num_heads=4, seq_len_q=16, seq_len_k=16, head_dim=32)
    output = scaled_dot_product_attention(q, k, v, causal=False)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("B,H,S,D", [(1, 1, 4, 8), (2, 4, 8, 16), (2, 8, 16, 32), (4, 8, 32, 64)])
@pytest.mark.parametrize("causal", [True, False])
def test_attention_matches_torch_multiple_shapes(B, H, S, D, causal):
    torch.manual_seed(42)
    q, k, v = make_qkv(batch_size=B, num_heads=H, seq_len_q=S, seq_len_k=S, head_dim=D)
    output = scaled_dot_product_attention(q, k, v, causal=causal)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_attention_different_sequence_lengths_non_causal():
    torch.manual_seed(42)
    B = 2
    H = 4
    Sq = 5
    Sk = 8
    D = 16
    q, k, v = make_qkv(batch_size=B, num_heads=H, seq_len_q=Sq, seq_len_k=Sk, head_dim=D)
    output = scaled_dot_product_attention(q, k, v, causal=False)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    assert output.shape == (B, H, Sq, D)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_causal_attention_does_not_see_future_v():
    torch.manual_seed(42)
    B = 1
    H = 2
    S = 8
    D = 16
    q, k, v = make_qkv(batch_size=B, num_heads=H, seq_len_q=S, seq_len_k=S, head_dim=D)
    output1 = scaled_dot_product_attention(q, k, v, causal=True)
    v_modified = v.clone()
    v_modified[..., -1, :] += 1000.0
    output2 = scaled_dot_product_attention(q, k, v_modified, causal=True)
    torch.testing.assert_close(output1[..., :-1, :], output2[..., :-1, :], rtol=1e-5, atol=1e-6)


def test_causal_attention_does_not_see_future_k():
    torch.manual_seed(42)
    B = 1
    H = 2
    S = 8
    D = 16
    q, k, v = make_qkv(batch_size=B, num_heads=H, seq_len_q=S, seq_len_k=S, head_dim=D)
    output1 = scaled_dot_product_attention(q, k, v, causal=True)
    k_modified = k.clone()
    k_modified[..., -1, :] += 1000.0
    output2 = scaled_dot_product_attention(q, k_modified, v, causal=True)
    torch.testing.assert_close(output1[..., :-1, :], output2[..., :-1, :], rtol=1e-5, atol=1e-6)


def test_non_causal_attention_can_see_future():
    torch.manual_seed(42)
    B = 1
    H = 1
    S = 4
    D = 8
    q, k, v = make_qkv(batch_size=B, num_heads=H, seq_len_q=S, seq_len_k=S, head_dim=D)
    output1 = scaled_dot_product_attention(q, k, v, causal=False)
    v_modified = v.clone()
    v_modified[..., -1, :] += 100.0
    output2 = scaled_dot_product_attention(q, k, v_modified, causal=False)
    assert not torch.allclose(output1[..., 0, :], output2[..., 0, :], rtol=1e-5, atol=1e-6)


def test_attention_scaling():
    B = 1
    H = 1
    S = 2
    D = 4
    q = torch.ones(B, H, S, D)
    k = torch.ones(B, H, S, D)
    v = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]])
    output = scaled_dot_product_attention(q, k, v, causal=False)
    expected = torch.tensor([[[[3.0, 4.0, 5.0, 6.0], [3.0, 4.0, 5.0, 6.0]]]])
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_attention_single_token():
    B = 2
    H = 4
    S = 1
    D = 8
    q, k, v = make_qkv(batch_size=B, num_heads=H, seq_len_q=S, seq_len_k=S, head_dim=D)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    torch.testing.assert_close(output, v, rtol=1e-5, atol=1e-6)


def test_attention_identical_keys_produces_uniform_attention():
    B = 1
    H = 1
    S = 4
    D = 8
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D)
    k_single = torch.randn(B, H, 1, D)
    k = k_single.expand(B, H, S, D)
    v = torch.randn(B, H, S, D)
    output = scaled_dot_product_attention(q, k, v, causal=False)
    expected = v.mean(dim=-2, keepdim=True).expand_as(v)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_attention_numerical_stability():
    B = 1
    H = 2
    S = 4
    D = 16
    q = torch.randn(B, H, S, D) * 100.0
    k = torch.randn(B, H, S, D) * 100.0
    v = torch.randn(B, H, S, D)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    assert torch.isfinite(output).all()


def test_attention_zero_inputs():
    B = 1
    H = 1
    S = 4
    D = 8
    q = torch.zeros(B, H, S, D)
    k = torch.zeros(B, H, S, D)
    v = torch.randn(B, H, S, D)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    expected = torch.zeros_like(v)
    for i in range(S):
        expected[..., i, :] = v[..., : i + 1, :].mean(dim=-2)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_attention_dtype(dtype):
    q, k, v = make_qkv(batch_size=2, num_heads=4, seq_len_q=8, seq_len_k=8, head_dim=16, dtype=dtype)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    assert output.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32]) # BF16 有一定的问题
def test_attention_cuda_dtype(dtype):
    device = torch.device("cuda")
    q, k, v = make_qkv(batch_size=2, num_heads=4, seq_len_q=16, seq_len_k=16, head_dim=32, dtype=dtype, device=device)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert output.dtype == dtype
    assert output.device == q.device

    if dtype == torch.float32:
        rtol = 1e-5
        atol = 1e-6
    elif dtype == torch.float16:
        rtol = 5e-3
        atol = 5e-3
    else:  # bfloat16
        rtol = 2e-2
        atol = 2e-2
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


def test_attention_backward():
    torch.manual_seed(42)
    q, k, v = make_qkv(batch_size=2, num_heads=4, seq_len_q=8, seq_len_k=8, head_dim=16, requires_grad=True)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    loss = output.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


def test_attention_gradient_matches_torch():
    torch.manual_seed(42)
    B = 2
    H = 2
    S = 8
    D = 16
    q1 = torch.randn(B, H, S, D, requires_grad=True)
    k1 = torch.randn(B, H, S, D, requires_grad=True)
    v1 = torch.randn(B, H, S, D, requires_grad=True)
    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)
    output1 = scaled_dot_product_attention(q1, k1, v1, causal=True)
    loss1 = output1.sum()
    loss1.backward()
    output2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
    loss2 = output2.sum()
    loss2.backward()
    torch.testing.assert_close(q1.grad, q2.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k1.grad, k2.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(v1.grad, v2.grad, rtol=1e-4, atol=1e-5)


def test_attention_rejects_invalid_q_dimension():
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, causal=True)


def test_attention_rejects_invalid_k_dimension():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 16)
    v = torch.randn(2, 4, 8, 16)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, causal=True)


def test_attention_rejects_invalid_v_dimension():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, causal=True)


def test_attention_rejects_mismatched_batch_size():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(3, 4, 8, 16)
    v = torch.randn(3, 4, 8, 16)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, causal=True)


def test_attention_rejects_mismatched_num_heads():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 8, 8, 16)
    v = torch.randn(2, 8, 8, 16)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, causal=True)


def test_attention_rejects_mismatched_k_v_sequence_length():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 16, 16)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, causal=True)


def test_attention_rejects_mismatched_q_k_head_dimension():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 32)
    v = torch.randn(2, 4, 8, 32)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, causal=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_attention_cuda():
    torch.manual_seed(42)
    device = torch.device("cuda")
    q, k, v = make_qkv(batch_size=2, num_heads=4, seq_len_q=16, seq_len_k=16, head_dim=32, dtype=torch.float32, device=device)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert output.device == q.device
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_attention_cuda_backward():
    torch.manual_seed(42)
    device = torch.device("cuda")
    q, k, v = make_qkv(batch_size=2, num_heads=4, seq_len_q=16, seq_len_k=16, head_dim=32, dtype=torch.float32, device=device, requires_grad=True)
    output = scaled_dot_product_attention(q, k, v, causal=True)
    loss = output.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()