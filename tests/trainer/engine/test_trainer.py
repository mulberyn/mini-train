import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from trainer.engine.trainer import Trainer


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 32)
        self.linear = nn.Linear(32, 32)

    def forward(self, x):
        return self.linear(self.embedding(x))


def make_loader():
    x = torch.randint(0, 32, (64, 8))
    y = torch.randint(0, 32, (64, 8))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8)


def test_trainer_runs():
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = make_loader()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        device="cpu",
        log_interval=1,
    )
    result = trainer.train(max_steps=5)
    assert "train_loss" in result
    assert np.isfinite(result["train_loss"])


def test_validation():
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = make_loader()
    valid_loader = make_loader()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device="cpu",
        eval_interval=2,
        eval_steps=2,
    )
    trainer.train(max_steps=2)
    metrics = trainer.evaluate()
    assert "valid/loss" in metrics
    assert "valid/ppl" in metrics
    assert metrics["valid/loss"] >= 0
    assert metrics["valid/ppl"] >= 1


def test_checkpoint(tmp_path):
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = make_loader()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    trainer.train(max_steps=2)
    latest = tmp_path / "latest.pt"
    step = tmp_path / "step_2.pt"
    assert latest.exists()
    assert step.exists()

    checkpoint = torch.load(step, map_location="cpu", weights_only=False)
    assert "model" in checkpoint
    assert "optimizer" in checkpoint
    assert checkpoint["step"] == 2
    assert "train_loss" in checkpoint

    model2 = TinyLM()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    trainer2 = Trainer(
        model=model2,
        optimizer=optimizer2,
        train_loader=loader,
        device="cpu",
    )
    trainer2.load_checkpoint(step)
    assert trainer2.global_step == 2