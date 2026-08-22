from pathlib import Path
import numpy as np
from trainer.tokenizer.bpe_tokenizer import BPETokenizer
from tqdm import tqdm   # 新增导入


def tokenize_file(
    input_file: str | Path,
    output_file: str | Path,
    tokenizer: BPETokenizer,
    dtype=np.uint16,
    chunk_size: int = 1_000_000,
):
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    buffer = []
    total_tokens = 0
    pbar = tqdm(desc="Tokenizing", unit="tokens")
    with input_file.open("r", encoding="utf-8") as fin, output_file.open("wb") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            pbar.update(len(ids))
            buffer.extend(ids)
            if len(buffer) >= chunk_size:
                array = np.asarray(buffer, dtype=dtype)
                array.tofile(fout)
                total_tokens += len(array)
                buffer.clear()
        if buffer:
            array = np.asarray(buffer, dtype=dtype)
            array.tofile(fout)
            total_tokens += len(array)
    pbar.close()
    return total_tokens