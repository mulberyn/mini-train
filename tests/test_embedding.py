import pytest
import torch
from torch import nn

from trainer.layers.embedding import Embedding


def test_embedding_shape():
    """Test the output shape of Embedding."""

    batch_size = 4
    seq_len = 16
    vocab_size = 1000
    embedding_dim = 64

    x = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
    )

    layer = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
    )

    y = layer(x)

    assert y.shape == (
        batch_size,
        seq_len,
        embedding_dim,
    )


def test_embedding_forward_matches_pytorch():
    """Test forward correctness against torch.nn.Embedding."""

    torch.manual_seed(42)

    vocab_size = 100
    embedding_dim = 32

    x = torch.tensor([
        [1, 5, 7, 10],
        [2, 5, 8, 10],
    ])

    reference = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
    )

    layer = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
    )

    # Make parameters identical.
    layer.weight.data.copy_(reference.weight.data)

    y = layer(x)
    y_ref = reference(x)

    torch.testing.assert_close(
        y,
        y_ref,
    )


def test_embedding_backward():
    """Test backward gradients against torch.nn.Embedding."""

    torch.manual_seed(42)

    vocab_size = 100
    embedding_dim = 32

    x = torch.tensor([
        [1, 5, 7, 10],
        [2, 5, 8, 10],
    ])

    reference = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
    )

    layer = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
    )

    layer.weight.data.copy_(reference.weight.data)

    y = layer(x)
    y_ref = reference(x)

    grad_output = torch.randn_like(y)

    y.backward(grad_output)
    y_ref.backward(grad_output)

    torch.testing.assert_close(
        layer.weight.grad,
        reference.weight.grad,
    )


def test_embedding_repeated_indices():
    """
    Test gradient accumulation when the same token appears
    multiple times.
    """

    vocab_size = 10
    embedding_dim = 8

    x = torch.tensor([
        [3, 3, 5, 7],
    ])

    layer = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
    )

    y = layer(x)

    loss = y.sum()
    loss.backward()

    # Token 3 appears twice, so its gradient should be
    # accumulated twice.
    expected = torch.full(
        (embedding_dim,),
        2.0,
    )

    torch.testing.assert_close(
        layer.weight.grad[3],
        expected,
    )

    # Token 5 appears once.
    expected = torch.ones(embedding_dim)

    torch.testing.assert_close(
        layer.weight.grad[5],
        expected,
    )


def test_embedding_dtype():
    """Test Embedding with different floating point dtypes."""

    for dtype in [
        torch.float32,
        torch.float64,
    ]:
        layer = Embedding(
            num_embeddings=100,
            embedding_dim=32,
            dtype=dtype,
        )

        x = torch.randint(
            0,
            100,
            (4, 16),
        )

        y = layer(x)

        assert y.dtype == dtype


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_embedding_cuda():
    """Test Embedding on CUDA."""

    device = torch.device("cuda")

    layer = Embedding(
        num_embeddings=1000,
        embedding_dim=64,
        device=device,
    )

    x = torch.randint(
        0,
        1000,
        (4, 16),
        device=device,
    )

    y = layer(x)

    assert y.device.type == "cuda"

    assert y.shape == (
        4,
        16,
        64,
    )


def test_embedding_parameter():
    """Test that embedding weight is a trainable parameter."""

    layer = Embedding(
        num_embeddings=1000,
        embedding_dim=64,
    )

    assert isinstance(
        layer.weight,
        nn.Parameter,
    )

    assert layer.weight.requires_grad

    assert layer.weight.shape == (
        1000,
        64,
    )