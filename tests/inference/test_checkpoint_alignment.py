"""End-to-end training → inference alignment test.

Validates the contract described in ``docs/generate_tinystories.md``:

- ``train_tinystories.py`` saves ``model_config.json`` before training and the
  ``Trainer`` writes checkpoints to ``artifacts/checkpoints/latest.pt`` with a
  ``"model"`` key.
- ``generate_tinystories.py`` / ``ModelRunner.from_checkpoint`` loads the same
  ``model_config.json`` and the same checkpoint and can generate text.

This test runs the real pipeline on synthetic data (tiny model, few steps).
"""

from pathlib import Path

import torch

from trainer.config import load_model_config, save_model_config
from trainer.data.dataloader import create_dataloader
from trainer.data.prepare import tokenize_file
from trainer.engine.trainer import Trainer
from trainer.model.transformer import TransformerLM
from trainer.optimizer.adamw import AdamW
from trainer.scheduler.lr_scheduler import LRScheduler
from trainer.tokenizer.bpe_tokenizer import BPETokenizer

from inference.model_runner import ModelRunner


CONTEXT_LENGTH = 32
D_MODEL = 32
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 64
ROPE_THETA = 10000.0
VOCAB_SIZE = 300


def _write_corpus(path: Path, repeat: int = 150):
    sentences = [
        "the cat sat on the mat",
        "a dog ran in the park",
        "tiny stories for tiny models",
        "once upon a time there was a bird",
    ]
    path.write_text("\n".join(sentences * repeat), encoding="utf-8")


def test_trainer_checkpoint_feeds_model_runner(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    checkpoints = artifacts / "checkpoints"
    train_text = tmp_path / "train.txt"
    valid_text = tmp_path / "valid.txt"
    _write_corpus(train_text)
    _write_corpus(valid_text)

    # --- training side ---------------------------------------------------
    tokenizer = BPETokenizer.train(
        files=[str(train_text)],
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer_path = artifacts / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    train_bin = artifacts / "train.bin"
    valid_bin = artifacts / "valid.bin"
    tokenize_file(train_text, train_bin, tokenizer)
    tokenize_file(valid_text, valid_bin, tokenizer)

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": CONTEXT_LENGTH,
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "d_ff": D_FF,
        "rope_theta": ROPE_THETA,
    }
    config_path = artifacts / "model_config.json"
    save_model_config(model_config, config_path)

    train_loader = create_dataloader(
        token_file=str(train_bin),
        seq_len=CONTEXT_LENGTH,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = create_dataloader(
        token_file=str(valid_bin),
        seq_len=CONTEXT_LENGTH,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = TransformerLM(**model_config, device="cpu", dtype=torch.float32)
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-4, warmup_steps=1, total_steps=10)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device="cpu",
        grad_clip=2.0,
        log_interval=1,
        eval_interval=2,
        eval_steps=1,
        checkpoint_dir=checkpoints,
        use_wandb=False,
    )
    trainer.train(max_steps=3)

    latest = checkpoints / "latest.pt"
    assert latest.exists(), "Trainer must write artifacts/checkpoints/latest.pt"
    assert (checkpoints / "step_3.pt").exists()

    # --- inference side --------------------------------------------------
    loaded_config = load_model_config(config_path)
    assert loaded_config == model_config

    runner = ModelRunner.from_checkpoint(
        checkpoint_path=str(latest),
        tokenizer_path=str(tokenizer_path),
        model_config=loaded_config,
        device="cpu",
    )
    assert runner.model.context_length == CONTEXT_LENGTH
    assert runner.model.vocab_size == tokenizer.vocab_size

    text = runner.generate_text("the cat", max_new_tokens=8, do_sample=False)
    assert isinstance(text, str)
    assert len(text) > 0

    # Streaming must produce the same text as the batched path.
    chunks = list(runner.stream_generate_text("the cat", max_new_tokens=8, do_sample=False))
    assert len(chunks) == 8
    assert "".join(chunks) == text[len("the cat"):]
