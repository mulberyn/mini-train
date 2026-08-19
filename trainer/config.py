"""Model config persistence.

``model_config.json`` is the contract between training and inference:

- ``train_tinystories.py`` saves the exact architecture parameters used to
  build ``TransformerLM`` before training starts.
- ``generate_tinystories.py`` / ``ModelRunner.from_checkpoint()`` load this
  same file instead of guessing hyper-parameters from the checkpoint.
"""

import json
from pathlib import Path
from typing import Any


def save_model_config(
    config: dict[str, Any],
    path: str | Path,
) -> None:
    """Serialize a model config dict to JSON.

    Args:
        config: Architecture parameters, e.g.
            ``{"vocab_size": ..., "context_length": ...,
               "d_model": ..., "num_layers": ..., "num_heads": ...,
               "d_ff": ..., "rope_theta": ...}``.
        path: Destination file, e.g. ``artifacts/model_config.json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def load_model_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load a model config JSON file.

    Args:
        path: Path to ``model_config.json``.

    Returns:
        The config dict as saved by :func:`save_model_config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
