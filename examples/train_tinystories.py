from pathlib import Path
import torch
from trainer.tokenizer.bpe_tokenizer import BPETokenizer
from trainer.data.prepare import tokenize_file
from trainer.data.dataloader import create_dataloader
from trainer.model.transformer import TransformerLM
from trainer.optimizer.adamw import AdamW
from trainer.scheduler.lr_scheduler import LRScheduler
from trainer.engine.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
TRAIN_TEXT = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VALID_TEXT = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
TOKENIZER_FILE = ARTIFACT_DIR / "tokenizer.json"
TRAIN_TOKENS = ARTIFACT_DIR / "train.bin"
VALID_TOKENS = ARTIFACT_DIR / "valid.bin"


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not TOKENIZER_FILE.exists():
        print("Training tokenizer...")
        tokenizer = BPETokenizer.train(
            files=[str(TRAIN_TEXT)],
            vocab_size=10_000,
            special_tokens=["<|endoftext|>"],
        )
        tokenizer.save(TOKENIZER_FILE)
    else:
        print("Loading tokenizer...")
        tokenizer = BPETokenizer.load(TOKENIZER_FILE)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    if not TRAIN_TOKENS.exists():
        print("Tokenizing training data...")
        tokenize_file(TRAIN_TEXT, TRAIN_TOKENS, tokenizer)
    if not VALID_TOKENS.exists():
        print("Tokenizing validation data...")
        tokenize_file(VALID_TEXT, VALID_TOKENS, tokenizer)

    context_length = 256
    batch_size = 32
    train_loader = create_dataloader(
        token_file=str(TRAIN_TOKENS),
        seq_len=context_length,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = create_dataloader(
        token_file=str(VALID_TOKENS),
        seq_len=context_length,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10_000,
        device=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    scheduler = LRScheduler(
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=500,
        total_steps=10_000,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        grad_clip=2.0,
        log_interval=10,
        eval_interval=500,
        eval_steps=20,
        checkpoint_dir=ARTIFACT_DIR / "checkpoints",
        use_wandb=True,
        wandb_project="miniLLM-engine",
        wandb_run_name="tinystories",
    )

    trainer.train(max_steps=10_000)


if __name__ == "__main__":
    main()