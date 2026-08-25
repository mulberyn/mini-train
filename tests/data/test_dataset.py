import numpy as np
import torch
from trainer.data.dataset import TokenizedDataset


def test_tokenized_dataset(tmp_path):
    token_file = tmp_path / "tokens.bin"
    tokens = np.arange(100, dtype=np.uint16)
    tokens.tofile(token_file)
    dataset = TokenizedDataset(token_file, seq_len=8)
    assert len(dataset) == 92
    x, y = dataset[0]
    assert x.shape == (8,)
    assert y.shape == (8,)
    assert torch.equal(x, torch.arange(0, 8))
    assert torch.equal(y, torch.arange(1, 9))


def test_dataset_random_access(tmp_path):
    token_file = tmp_path / "tokens.bin"
    tokens = np.arange(1000, dtype=np.uint16)
    tokens.tofile(token_file)
    dataset = TokenizedDataset(token_file, seq_len=32)
    x, y = dataset[100]
    assert x[0].item() == 100
    assert y[0].item() == 101
    assert x[-1].item() == 131
    assert y[-1].item() == 132