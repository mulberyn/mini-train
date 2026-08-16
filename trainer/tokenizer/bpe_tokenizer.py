from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers, decoders


class BPETokenizer:
    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer


    @classmethod
    def train(cls, files: list[str], vocab_size: int, special_tokens: list[str]):
        tokenizer = HFTokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, show_progress=True)
        tokenizer.train(files=files, trainer=trainer)
        return cls(tokenizer)


    @classmethod
    def load(cls, path: str | Path):
        tokenizer = HFTokenizer.from_file(str(path))
        return cls(tokenizer)


    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))


    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids


    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [encoding.ids for encoding in self.tokenizer.encode_batch(texts)]


    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Token {token!r} does not exist in tokenizer.")
        return token_id


    def id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id)


    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


    def __len__(self):
        return self.vocab_size