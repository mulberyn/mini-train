from pathlib import Path
import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler=None,
    step: int = 0,
    best_val_loss: float | None = None,
    train_loss: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "train_loss": train_loss,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return {
        "step": checkpoint["step"],
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "train_loss": checkpoint.get("train_loss"),
    }
