import pytest
import torch
from torch import nn

from trainer.utils.gradient_clip import gradient_clipping


def test_no_gradients():
    p1 = nn.Parameter(torch.randn(10))
    p2 = nn.Parameter(torch.randn(20))

    total_norm = gradient_clipping([p1, p2], max_norm=1.0)

    assert total_norm == 0.0


def test_l2_norm():
    p1 = nn.Parameter(torch.tensor([3.0, 4.0]))
    p1.grad = torch.tensor([3.0, 4.0])

    total_norm = gradient_clipping([p1], max_norm=100.0)

    assert total_norm == pytest.approx(5.0)


def test_global_gradient_norm():
    p1 = nn.Parameter(torch.zeros(2))
    p2 = nn.Parameter(torch.zeros(2))

    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = torch.tensor([12.0, 5.0])

    total_norm = gradient_clipping([p1, p2], max_norm=100.0)

    expected = torch.sqrt(torch.tensor(3**2 + 4**2 + 12**2 + 5**2)).item()
    assert total_norm == pytest.approx(expected)


def test_gradient_is_clipped():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    p.grad = torch.tensor([3.0, 4.0])

    total_norm = gradient_clipping([p], max_norm=1.0)

    assert total_norm == pytest.approx(5.0)

    new_norm = torch.linalg.vector_norm(p.grad).item()
    assert new_norm == pytest.approx(1.0)

    expected = torch.tensor([0.6, 0.8])
    torch.testing.assert_close(p.grad, expected, rtol=1e-6, atol=1e-6)


def test_gradient_not_clipped_when_below_threshold():
    p = nn.Parameter(torch.tensor([0.3, 0.4]))
    p.grad = torch.tensor([0.3, 0.4])
    original = p.grad.clone()

    total_norm = gradient_clipping([p], max_norm=1.0)

    assert total_norm == pytest.approx(0.5)
    torch.testing.assert_close(p.grad, original)


def test_multiple_parameters_are_scaled_equally():
    p1 = nn.Parameter(torch.zeros(2))
    p2 = nn.Parameter(torch.zeros(2))

    p1.grad = torch.tensor([3.0, 0.0])
    p2.grad = torch.tensor([4.0, 0.0])

    total_norm = gradient_clipping([p1, p2], max_norm=1.0)

    assert total_norm == pytest.approx(5.0)

    torch.testing.assert_close(p1.grad, torch.tensor([0.6, 0.0]))
    torch.testing.assert_close(p2.grad, torch.tensor([0.8, 0.0]))


def test_none_gradients_are_ignored():
    p1 = nn.Parameter(torch.randn(4))
    p2 = nn.Parameter(torch.randn(4))

    p1.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])
    p2.grad = None

    total_norm = gradient_clipping([p1, p2], max_norm=1.0)

    assert total_norm == pytest.approx(5.0)
    assert p2.grad is None


def test_l1_norm():
    p = nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([-3.0, 4.0, -5.0])

    total_norm = gradient_clipping([p], max_norm=100.0, norm_type=1.0)

    assert total_norm == pytest.approx(12.0)


def test_l1_gradient_clipping():
    p = nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([3.0, 4.0, 5.0])

    gradient_clipping([p], max_norm=6.0, norm_type=1.0)

    expected = torch.tensor([1.5, 2.0, 2.5])
    torch.testing.assert_close(p.grad, expected)


def test_inf_norm():
    p = nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([-1.0, 3.0, -7.0, 2.0])

    total_norm = gradient_clipping([p], max_norm=100.0, norm_type=float("inf"))

    assert total_norm == pytest.approx(7.0)


def test_inf_norm_clipping():
    p = nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([2.0, 4.0, 8.0, 1.0])

    gradient_clipping([p], max_norm=4.0, norm_type=float("inf"))

    expected = torch.tensor([1.0, 2.0, 4.0, 0.5])
    torch.testing.assert_close(p.grad, expected)


@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
def test_matches_torch(norm_type):
    torch.manual_seed(42)

    p1 = nn.Parameter(torch.randn(16))
    p2 = nn.Parameter(torch.randn(32))

    p1.grad = torch.randn_like(p1)
    p2.grad = torch.randn_like(p2)

    p1_ref = nn.Parameter(p1.detach().clone())
    p2_ref = nn.Parameter(p2.detach().clone())
    p1_ref.grad = p1.grad.clone()
    p2_ref.grad = p2.grad.clone()

    max_norm = 1.0

    expected_norm = torch.nn.utils.clip_grad_norm_(
        [p1_ref, p2_ref],
        max_norm=max_norm,
        norm_type=norm_type,
    )

    actual_norm = gradient_clipping(
        [p1, p2],
        max_norm=max_norm,
        norm_type=norm_type,
    )

    assert actual_norm == pytest.approx(expected_norm.item(), rel=1e-5, abs=1e-6)

    torch.testing.assert_close(p1.grad, p1_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(p2.grad, p2_ref.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gradient_clipping_cuda():
    device = torch.device("cuda")
    p = nn.Parameter(torch.randn(1024, device=device))
    p.grad = torch.randn_like(p)

    total_norm = gradient_clipping([p], max_norm=1.0)

    assert isinstance(total_norm, float)
    assert p.grad.device.type == "cuda"

    actual_norm = torch.linalg.vector_norm(p.grad).item()
    assert actual_norm <= 1.0 + 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gradient_clipping_cuda_multiple_parameters():
    device = torch.device("cuda")

    parameters = [nn.Parameter(torch.randn(128, device=device)) for _ in range(8)]

    for p in parameters:
        p.grad = torch.randn_like(p)

    total_norm = gradient_clipping(parameters, max_norm=1.0)

    assert isinstance(total_norm, float)

    global_norm = torch.sqrt(
        sum(torch.sum(p.grad**2) for p in parameters)
    )
    assert global_norm.item() <= 1.0 + 1e-5


def test_accepts_parameter_generator():
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))

    for p in model.parameters():
        p.grad = torch.randn_like(p)

    total_norm = gradient_clipping(model.parameters(), max_norm=1.0)

    assert isinstance(total_norm, float)


def test_invalid_max_norm():
    p = nn.Parameter(torch.randn(4))
    p.grad = torch.randn_like(p)

    with pytest.raises(ValueError):
        gradient_clipping([p], max_norm=0.0)


def test_invalid_norm_type():
    p = nn.Parameter(torch.randn(4))
    p.grad = torch.randn_like(p)

    with pytest.raises(ValueError):
        gradient_clipping([p], max_norm=1.0, norm_type=0.0)


def test_gradient_clipping_training_loop():
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(10):
        x = torch.randn(8, 32)
        y = torch.randn(8, 16)

        optimizer.zero_grad()

        pred = model(x)
        loss = ((pred - y) ** 2).mean()

        loss.backward()

        total_norm = gradient_clipping(model.parameters(), max_norm=1.0)

        assert isinstance(total_norm, float)

        optimizer.step()