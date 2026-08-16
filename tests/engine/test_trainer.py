import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainer.engine.trainer import Trainer
from trainer.optimizer.adamw import AdamW
from trainer.scheduler.lr_scheduler import LRScheduler


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


def make_trainer(device="cpu", max_grad_norm=None, grad_clip_norm_type=2.0):
    model = TinyLM()
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = LRScheduler(
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=2,
        total_steps=10,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device(device),
        max_grad_norm=max_grad_norm,
        grad_clip_norm_type=grad_clip_norm_type,
        log_interval=1,
    )
    return model, optimizer, scheduler, trainer


def test_train_step_returns_stats():
    _, _, _, trainer = make_trainer()
    input_ids, labels = make_batch()
    stats = trainer.train_step(input_ids, labels)
    assert stats.step == 1
    assert isinstance(stats.loss, float)
    assert isinstance(stats.grad_norm, float)
    assert stats.loss > 0


def test_parameters_are_updated():
    model, _, _, trainer = make_trainer()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    input_ids, labels = make_batch()
    trainer.train_step(input_ids, labels)
    changed = False
    for name, param in model.named_parameters():
        if not torch.equal(before[name], param):
            changed = True
            break
    assert changed


def test_global_step_increments():
    _, _, _, trainer = make_trainer()
    input_ids, labels = make_batch()
    assert trainer.global_step == 0
    trainer.train_step(input_ids, labels)
    assert trainer.global_step == 1
    trainer.train_step(input_ids, labels)
    assert trainer.global_step == 2


def test_gradients_are_computed():
    model, _, _, trainer = make_trainer()
    input_ids, labels = make_batch()
    trainer.train_step(input_ids, labels)
    for param in model.parameters():
        assert param.grad is not None


def test_gradient_clipping():
    model, _, _, trainer = make_trainer(max_grad_norm=0.1)
    input_ids, labels = make_batch()
    stats = trainer.train_step(input_ids, labels)
    assert stats.grad_norm >= 0.0
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += (param.grad.detach() ** 2).sum().item()
    total_norm = total_norm ** 0.5
    assert total_norm <= 0.1 + 1e-5


def test_without_gradient_clipping():
    _, _, _, trainer = make_trainer(max_grad_norm=None)
    input_ids, labels = make_batch()
    stats = trainer.train_step(input_ids, labels)
    assert stats.grad_norm == 0.0


def test_scheduler_updates():
    _, optimizer, scheduler, trainer = make_trainer()
    input_ids, labels = make_batch()
    initial_lr = optimizer.param_groups[0]["lr"]
    trainer.train_step(input_ids, labels)
    assert scheduler.current_step == 1
    current_lr = optimizer.param_groups[0]["lr"]
    assert current_lr != initial_lr


def test_training_loss_decreases():
    torch.manual_seed(42)
    model = TinyLM(vocab_size=8, d_model=16)
    optimizer = AdamW(
        model.parameters(),
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        max_grad_norm=1.0,
    )
    input_ids = torch.zeros(16, 8, dtype=torch.long)
    labels = torch.zeros(16, 8, dtype=torch.long)
    losses = []
    for _ in range(100):
        stats = trainer.train_step(input_ids, labels)
        losses.append(stats.loss)
    assert losses[-1] < losses[0]


def test_fit():
    _, _, _, trainer = make_trainer()
    batches = [make_batch() for _ in range(4)]
    stats = trainer.fit(batches, num_steps=8)
    assert len(stats) == 8
    assert stats[-1].step == 8
    for stat in stats:
        assert isinstance(stat.loss, float)


def test_fit_with_dataloader():
    input_ids = torch.randint(0, 32, (32, 8))
    labels = torch.randint(0, 32, (32, 8))
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    _, _, _, trainer = make_trainer()
    stats = trainer.fit(dataloader, num_steps=10)
    assert len(stats) == 10
    assert stats[-1].step == 10


def test_optimizer_state_created():
    model, optimizer, _, trainer = make_trainer()
    input_ids, labels = make_batch()
    trainer.train_step(input_ids, labels)
    for param in model.parameters():
        if param.requires_grad:
            state = optimizer.state[param]
            assert "m" in state
            assert "v" in state
            assert "step" in state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_trainer_cuda():
    device = torch.device("cuda")
    model, _, _, trainer = make_trainer(device=device)
    input_ids, labels = make_batch(device=device)
    stats = trainer.train_step(input_ids, labels)
    assert isinstance(stats.loss, float)
    for param in model.parameters():
        assert param.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_trainer_cuda_gradients():
    device = torch.device("cuda")
    model, _, _, trainer = make_trainer(device=device)
    input_ids, labels = make_batch(device=device)
    trainer.train_step(input_ids, labels)
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert param.grad.device.type == "cuda"