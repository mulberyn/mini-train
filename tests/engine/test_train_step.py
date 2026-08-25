import pytest
import torch
from torch import nn
from trainer.engine.train_step import TrainStep
from trainer.optimizer.adamw import AdamW
from trainer.utils.lr_scheduler import LRScheduler


class TinyLM(nn.Module):
    def __init__(self, vocab_size=32, d_model=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.linear(x)


def make_batch(batch_size=4, seq_len=8, vocab_size=32, device="cpu"):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids, labels


def make_train_step(device="cpu"):
    model = TinyLM().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-4, warmup_steps=10, total_steps=100)
    train_step = TrainStep(model=model, optimizer=optimizer, scheduler=scheduler)
    return model, optimizer, scheduler, train_step


def test_train_step_returns_float():
    model, optimizer, scheduler, train_step = make_train_step()
    input_ids, labels = make_batch()
    loss = train_step.step(input_ids, labels)
    assert isinstance(loss, float)
    assert loss > 0.0


def test_train_step_forward_backward():
    model, optimizer, scheduler, train_step = make_train_step()
    input_ids, labels = make_batch()
    train_step.step(input_ids, labels)
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def test_parameters_are_updated():
    model, optimizer, scheduler, train_step = make_train_step()
    input_ids, labels = make_batch()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    train_step.step(input_ids, labels)
    after = {name: param.detach().clone() for name, param in model.named_parameters()}
    changed = [not torch.equal(before[name], after[name]) for name in before]
    assert any(changed)


def test_optimizer_state_is_initialized():
    model, optimizer, scheduler, train_step = make_train_step()
    input_ids, labels = make_batch()
    assert len(optimizer.state) == 0
    train_step.step(input_ids, labels)
    assert len(optimizer.state) > 0


def test_scheduler_is_updated():
    model, optimizer, scheduler, train_step = make_train_step()
    input_ids, labels = make_batch()
    assert scheduler.current_step == 0
    train_step.step(input_ids, labels)
    assert scheduler.current_step == 1


def test_scheduler_is_not_required():
    model = TinyLM()
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    train_step = TrainStep(model=model, optimizer=optimizer, scheduler=None)
    input_ids, labels = make_batch()
    loss = train_step.step(input_ids, labels)
    assert isinstance(loss, float)


def test_gradients_are_recomputed_each_step():
    model, optimizer, scheduler, train_step = make_train_step()
    input_ids, labels = make_batch()
    train_step.step(input_ids, labels)
    first_gradients = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    train_step.step(input_ids, labels)
    second_gradients = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    for name in first_gradients:
        assert not torch.equal(first_gradients[name], second_gradients[name])


@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (1, 8), (2, 16), (4, 32)])
def test_train_step_different_shapes(batch_size, seq_len):
    model, optimizer, scheduler, train_step = make_train_step()
    input_ids, labels = make_batch(batch_size=batch_size, seq_len=seq_len)
    loss = train_step.step(input_ids, labels)
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_loss_decreases():
    torch.manual_seed(42)
    model = TinyLM(vocab_size=16, d_model=32)
    optimizer = AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    train_step = TrainStep(model=model, optimizer=optimizer)
    input_ids, labels = make_batch(batch_size=4, seq_len=8, vocab_size=16)
    losses = []
    for _ in range(100):
        loss = train_step.step(input_ids, labels)
        losses.append(loss)
    assert losses[-1] < losses[0]


def test_loss_is_finite():
    model = TinyLM()
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    train_step = TrainStep(model=model, optimizer=optimizer)
    input_ids, labels = make_batch()
    loss = train_step.step(input_ids, labels)
    assert torch.isfinite(torch.tensor(loss))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_train_step_cuda():
    device = torch.device("cuda")
    model, optimizer, scheduler, train_step = make_train_step(device=device)
    input_ids, labels = make_batch(device=device)
    loss = train_step.step(input_ids, labels)
    assert isinstance(loss, float)
    for param in model.parameters():
        assert param.device.type == device.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_train_step_cuda_gradients():
    device = torch.device("cuda")
    model, optimizer, scheduler, train_step = make_train_step(device=device)
    input_ids, labels = make_batch(device=device)
    train_step.step(input_ids, labels)
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert param.device.type == device.type
            assert torch.isfinite(param.grad).all()