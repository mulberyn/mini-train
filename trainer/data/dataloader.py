from torch.utils.data import DataLoader
from .dataset import TokenizedDataset


def create_dataloader(
    token_file: str,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
):
    dataset = TokenizedDataset(token_file=token_file, seq_len=seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader