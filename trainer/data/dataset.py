from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    def __init__(
        self,
        token_file: str | Path,
        seq_len: int,
        dtype=np.uint16,
    ):
        self.token_file = Path(token_file)
        self.seq_len = seq_len
        if not self.token_file.exists():
            raise FileNotFoundError(self.token_file)
        self.tokens = np.memmap(self.token_file, mode="r", dtype=dtype)
        if len(self.tokens) <= seq_len:
            raise ValueError("Dataset must contain more tokens than seq_len.")


    def __len__(self):
        return len(self.tokens) - self.seq_len


    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + self.seq_len + 1]
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        return x, y