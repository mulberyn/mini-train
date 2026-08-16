import json
from trainer.tokenizer.bpe_tokenizer import BPETokenizer


def test_train_encode_decode(tmp_path):
    train_file = tmp_path / "train.txt"
    train_file.write_text(
        """
        Once upon a time there was a little cat.
        The cat liked to play.
        Once upon a time there was a little dog.
        The dog liked to run.
        """,
        encoding="utf-8",
    )
    tokenizer = BPETokenizer.train(
        files=[str(train_file)],
        vocab_size=100,
        special_tokens=["<|endoftext|>"],
    )
    assert tokenizer.vocab_size <= 100
    ids = tokenizer.encode("Once upon a time.")
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(x, int) for x in ids)


def test_special_token(tmp_path):
    train_file = tmp_path / "train.txt"
    train_file.write_text("hello world\nhello world\n", encoding="utf-8")
    tokenizer = BPETokenizer.train(
        files=[str(train_file)],
        vocab_size=50,
        special_tokens=["<|endoftext|>"],
    )
    token_id = tokenizer.token_to_id("<|endoftext|>")
    assert isinstance(token_id, int)
    ids = tokenizer.encode("hello <|endoftext|> world")
    assert token_id in ids


def test_save_and_load(tmp_path):
    train_file = tmp_path / "train.txt"
    tokenizer_file = tmp_path / "tokenizer.json"
    train_file.write_text("hello world\nhello world\n", encoding="utf-8")
    tokenizer = BPETokenizer.train(
        files=[str(train_file)],
        vocab_size=50,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.save(tokenizer_file)
    loaded = BPETokenizer.load(tokenizer_file)
    text = "hello world"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == tokenizer.decode(tokenizer.encode(text))


def test_encode_batch(tmp_path):
    train_file = tmp_path / "train.txt"
    train_file.write_text("hello world\nhello there\n", encoding="utf-8")
    tokenizer = BPETokenizer.train(
        files=[str(train_file)],
        vocab_size=50,
        special_tokens=["<|endoftext|>"],
    )
    result = tokenizer.encode_batch(["hello world", "hello there"])
    assert len(result) == 2
    assert all(isinstance(x, list) for x in result)