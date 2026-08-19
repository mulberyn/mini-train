from pathlib import Path
import json
import pytest
from trainer.config import save_model_config, load_model_config


MODEL_CONFIG = {
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 1344,
    "rope_theta": 10000.0,
}


def test_save_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "model_config.json"
    save_model_config(MODEL_CONFIG, path)
    assert path.exists()
    loaded = load_model_config(path)
    assert loaded == MODEL_CONFIG


def test_saved_file_is_readable_json(tmp_path: Path):
    path = tmp_path / "model_config.json"
    save_model_config(MODEL_CONFIG, path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw == MODEL_CONFIG


def test_save_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "model_config.json"
    save_model_config(MODEL_CONFIG, path)
    assert path.exists()
    assert load_model_config(path) == MODEL_CONFIG


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model_config(tmp_path / "does_not_exist.json")
