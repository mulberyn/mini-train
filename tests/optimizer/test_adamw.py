import pytest
import torch
from torch import nn
from trainer.optimizer.adamw import AdamW


def make_optimizer(model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def test_adamw_initialization():
    model = nn.Linear(10, 5)
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == 1e-3
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.01


@pytest.mark.parametrize("kwargs", [{"lr": 0.0}, {"lr": -1e-3}, {"betas": (-0.1, 0.999)}, {"betas": (0.9, 1.0)}, {"eps": 0.0}, {"eps": -1e-8}, {"weight_decay": -0.1}])
def test_adamw_invalid_arguments(kwargs):
    model = nn.Linear(4, 4)
    with pytest.raises((ValueError, AssertionError)):
        AdamW(model.parameters(), **kwargs)


def test_single_step():
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamW([param], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    param.grad = torch.tensor([1.0])
    optimizer.step()
    torch.testing.assert_close(param, torch.tensor([0.9]), rtol=1e-6, atol=1e-6)


def test_bias_correction():
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamW([param], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    param.grad = torch.tensor([1.0])
    optimizer.step()
    state = optimizer.state[param]
    assert state["step"] == 1
    torch.testing.assert_close(state["m"], torch.tensor([0.1]))
    torch.testing.assert_close(state["v"], torch.tensor([0.001]))


def test_decoupled_weight_decay():
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamW([param], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    param.grad = torch.tensor([0.0])
    optimizer.step()
    torch.testing.assert_close(param, torch.tensor([0.99]), rtol=1e-6, atol=1e-6)


def test_weight_decay_is_decoupled_from_gradient():
    p1 = nn.Parameter(torch.tensor([1.0]))
    p2 = nn.Parameter(torch.tensor([1.0]))
    opt1 = AdamW([p1], lr=0.1, weight_decay=0.1)
    opt2 = AdamW([p2], lr=0.1, weight_decay=0.0)
    p1.grad = torch.tensor([1.0])
    p2.grad = torch.tensor([1.0])
    opt1.step()
    opt2.step()
    assert p1.item() < p2.item()


def test_multiple_steps():
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamW([param], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for _ in range(10):
        param.grad = torch.tensor([1.0])
        optimizer.step()
    assert optimizer.state[param]["step"] == 10
    assert param.item() < 1.0


def test_none_gradient_is_skipped():
    p1 = nn.Parameter(torch.tensor([1.0]))
    p2 = nn.Parameter(torch.tensor([2.0]))
    optimizer = AdamW([p1, p2], lr=0.1)
    p1.grad = torch.tensor([1.0])
    p2.grad = None
    p2_before = p2.detach().clone()
    optimizer.step()
    assert optimizer.state[p1]["step"] == 1
    assert p2 not in optimizer.state
    torch.testing.assert_close(p2, p2_before)


def test_zero_gradient():
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamW([param], lr=0.1, weight_decay=0.0)
    param.grad = torch.zeros_like(param)
    before = param.detach().clone()
    optimizer.step()
    torch.testing.assert_close(param, before)


@pytest.mark.parametrize("shape", [(1,), (4,), (4, 4), (2, 3, 4)])
def test_parameter_shapes(shape):
    param = nn.Parameter(torch.randn(shape))
    optimizer = AdamW([param], lr=1e-3)
    param.grad = torch.randn_like(param)
    before = param.detach().clone()
    optimizer.step()
    assert param.shape == shape
    assert not torch.equal(param, before)


def test_matches_torch_adamw():
    torch.manual_seed(42)
    p1 = nn.Parameter(torch.randn(8, 8))
    p2 = nn.Parameter(p1.detach().clone())
    optimizer_custom = AdamW([p1], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    optimizer_torch = torch.optim.AdamW([p2], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    for _ in range(10):
        grad = torch.randn_like(p1)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        optimizer_custom.step()
        optimizer_torch.step()
    torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-6)


def test_parameter_groups():
    p1 = nn.Parameter(torch.ones(4))
    p2 = nn.Parameter(torch.ones(4))
    optimizer = AdamW([{"params": [p1], "lr": 1e-2}, {"params": [p2], "lr": 1e-3}])
    p1.grad = torch.ones_like(p1)
    p2.grad = torch.ones_like(p2)
    optimizer.step()
    assert p1.mean() < p2.mean()


def test_closure():
    param = nn.Parameter(torch.tensor([2.0]))
    optimizer = AdamW([param], lr=0.1)

    def closure():
        optimizer.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None
    assert torch.isfinite(loss)


def test_optimizer_state_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param = nn.Parameter(torch.randn(16, device=device))
    optimizer = AdamW([param], lr=1e-3)
    param.grad = torch.randn_like(param)
    optimizer.step()
    state = optimizer.state[param]
    assert state["m"].device == param.device
    assert state["v"].device == param.device


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_optimizer_dtype(dtype):
    param = nn.Parameter(torch.randn(16, dtype=dtype))
    optimizer = AdamW([param], lr=1e-3)
    param.grad = torch.randn_like(param)
    optimizer.step()
    assert param.dtype == dtype
    assert optimizer.state[param]["m"].dtype == dtype
    assert optimizer.state[param]["v"].dtype == dtype


def test_linear_regression_matches_torch():
    torch.manual_seed(42)
    x = torch.randn(128, 4)
    true_w = torch.tensor([[2.0], [-1.0], [0.5], [3.0]])
    y = x @ true_w
    model_custom = nn.Linear(4, 1)
    model_torch = nn.Linear(4, 1)
    model_torch.load_state_dict(model_custom.state_dict())
    optimizer_custom = AdamW(model_custom.parameters(), lr=1e-2, weight_decay=0.0)
    optimizer_torch = torch.optim.AdamW(model_torch.parameters(), lr=1e-2, weight_decay=0.0)
    for step in range(500):
        pred_custom = model_custom(x)
        loss_custom = ((pred_custom - y) ** 2).mean()
        pred_torch = model_torch(x)
        loss_torch = ((pred_torch - y) ** 2).mean()
        optimizer_custom.zero_grad()
        loss_custom.backward()
        optimizer_custom.step()
        optimizer_torch.zero_grad()
        loss_torch.backward()
        optimizer_torch.step()
    torch.testing.assert_close(model_custom.weight, model_torch.weight, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model_custom.bias, model_torch.bias, rtol=1e-5, atol=1e-6)


def test_matches_torch_adamw_step_by_step():
    torch.manual_seed(42)
    p_custom = nn.Parameter(torch.randn(8, 8))
    p_torch = nn.Parameter(p_custom.detach().clone())
    optimizer_custom = AdamW([p_custom], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    optimizer_torch = torch.optim.AdamW([p_torch], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for step in range(10):
        grad = torch.randn_like(p_custom)
        p_custom.grad = grad.clone()
        p_torch.grad = grad.clone()
        optimizer_custom.step()
        optimizer_torch.step()
        torch.testing.assert_close(p_custom, p_torch, rtol=1e-5, atol=1e-6)