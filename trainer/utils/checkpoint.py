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
):
    """Save a full training checkpoint.

    The format is the shared contract consumed by ``ModelRunner``:

    .. code-block:: python

        {
            "step": ...,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),  # optional
            "best_val_loss": ...,
            "train_loss": ...,
        }

    Args:
        path: Destination file, e.g. ``artifacts/checkpoints/latest.pt``.
        model: Model whose ``state_dict()`` is saved under the ``"model"`` key.
        optimizer: Optimizer whose ``state_dict()`` is saved.
        scheduler: Optional scheduler; saved when not ``None``.
        step: Global training step.
        best_val_loss: Best validation loss seen so far.
        train_loss: Latest training loss.
    """
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
    """Load a checkpoint saved by :func:`save_checkpoint` into model/optimizer.

    Args:
        path: Checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.

    Returns:
        A dict with ``"step"``, ``"best_val_loss"`` and ``"train_loss"``.
    """
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
