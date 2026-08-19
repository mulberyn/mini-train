from pathlib import Path
import pytest
import torch
from inference.model_runner import ModelRunner
from trainer.config import save_model_config
from trainer.model.transformer import TransformerLM
from trainer.tokenizer.bpe_tokenizer import BPETokenizer as Tokenizer


VOCAB_SIZE = 128
CONTEXT_LENGTH = 32
D_MODEL = 32
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 64
ROPE_THETA = 10000.0

MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "context_length": CONTEXT_LENGTH,
    "d_model": D_MODEL,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "d_ff": D_FF,
    "rope_theta": ROPE_THETA,
}


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % VOCAB_SIZE for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(int(i)) for i in ids)


def make_model(device="cpu"):
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        device=device,
        dtype=torch.float32,
    )
    return model


def make_runner(device="cpu"):
    model = make_model(device)
    tokenizer = DummyTokenizer()
    return ModelRunner(model=model, tokenizer=tokenizer, device=device)


def test_runner_initialization():
    runner = make_runner()
    assert runner.model.training is False
    assert runner.device == torch.device("cpu")


def test_model_is_eval():
    runner = make_runner()
    assert not runner.model.training


def test_model_parameters_on_device():
    runner = make_runner()
    for parameter in runner.model.parameters():
        assert parameter.device.type == "cpu"


def test_encode():
    runner = make_runner()
    tokens = runner.encode("abc")
    assert isinstance(tokens, list)
    assert len(tokens) == 3
    assert all(isinstance(x, int) for x in tokens)


def test_decode():
    runner = make_runner()
    ids = [97, 98, 99]
    text = runner.decode(ids)
    assert text == "abc"


def test_decode_tensor():
    # Regression: ModelRunner.decode must accept a torch.Tensor (e.g. a row
    # of the tensor returned by ModelRunner.generate) and convert it to ids.
    runner = make_runner()
    ids = torch.tensor([97, 98, 99])
    text = runner.decode(ids)
    assert text == "abc"


def test_decode_tensor_on_device():
    runner = make_runner()
    ids = torch.tensor([97, 98, 99])
    text = runner.decode(ids.to(runner.device))
    assert text == "abc"


def test_encode_decode():
    runner = make_runner()
    text = "hello"
    ids = runner.encode(text)
    decoded = runner.decode(ids)
    assert decoded == text


def test_forward_shape():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
    logits = runner.forward(input_ids)
    assert logits.shape == (2, 8, VOCAB_SIZE)


def test_forward_dtype():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
    logits = runner.forward(input_ids)
    assert logits.dtype == torch.float32


def test_forward_device():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
    logits = runner.forward(input_ids)
    assert logits.device.type == "cpu"


def test_forward_rejects_invalid_shape():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (8,))
    with pytest.raises(ValueError):
        runner.forward(input_ids)


def test_greedy_sampling():
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0], [8.0, 2.0, 1.0, 0.0]])
    tokens = ModelRunner._sample_next_token(logits, do_sample=False)
    expected = torch.tensor([1, 0])
    torch.testing.assert_close(tokens, expected)


def test_temperature_changes_distribution():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    low_temperature = ModelRunner._apply_temperature(logits, temperature=0.5)
    high_temperature = ModelRunner._apply_temperature(logits, temperature=2.0)
    assert torch.allclose(low_temperature, logits / 0.5)
    assert torch.allclose(high_temperature, logits / 2.0)


def test_temperature_must_be_positive():
    logits = torch.randn(2, 8)
    with pytest.raises(ValueError):
        ModelRunner._apply_temperature(logits, temperature=0.0)


def test_top_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    filtered = ModelRunner._top_k(logits, k=2)
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 3])
    assert filtered[0, 1] == 5.0
    assert filtered[0, 2] == 3.0


def test_top_k_larger_than_vocab():
    logits = torch.randn(2, 8)
    output = ModelRunner._top_k(logits, k=100)
    torch.testing.assert_close(output, logits)


def test_top_k_invalid():
    logits = torch.randn(2, 8)
    with pytest.raises(ValueError):
        ModelRunner._top_k(logits, k=0)


def test_generate_shape():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
    output = runner.generate(input_ids, max_new_tokens=5, do_sample=False)
    assert output.shape == (2, 13)


def test_generate_does_not_modify_input():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    original = input_ids.clone()
    runner.generate(input_ids, max_new_tokens=5, do_sample=False)
    torch.testing.assert_close(input_ids, original)


def test_generate_zero_tokens():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    output = runner.generate(input_ids, max_new_tokens=0)
    torch.testing.assert_close(output, input_ids)


def test_greedy_generation_is_deterministic():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    output1 = runner.generate(input_ids, max_new_tokens=8, do_sample=False)
    output2 = runner.generate(input_ids, max_new_tokens=8, do_sample=False)
    torch.testing.assert_close(output1, output2)


def test_sampling_with_same_seed_is_deterministic():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    torch.manual_seed(42)
    output1 = runner.generate(input_ids, max_new_tokens=8, temperature=1.0, do_sample=True)
    torch.manual_seed(42)
    output2 = runner.generate(input_ids, max_new_tokens=8, temperature=1.0, do_sample=True)
    torch.testing.assert_close(output1, output2)


def test_generate_respects_context_length():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, CONTEXT_LENGTH))
    output = runner.generate(input_ids, max_new_tokens=10, do_sample=False)
    assert output.shape == (1, CONTEXT_LENGTH + 10)


def test_stream_generate_yields_new_tokens():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    tokens = list(runner.stream_generate(input_ids, max_new_tokens=5, do_sample=False))
    assert len(tokens) == 5
    assert all(token.shape == (1, 1) for token in tokens)
    assert all(token.dtype == torch.long for token in tokens)


def test_stream_generate_matches_generate():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    tokens = list(runner.stream_generate(input_ids, max_new_tokens=8, do_sample=False))
    full = runner.generate(input_ids, max_new_tokens=8, do_sample=False)
    rebuilt = torch.cat([input_ids, *tokens], dim=-1)
    torch.testing.assert_close(rebuilt, full)


def test_stream_generate_zero_tokens():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    assert list(runner.stream_generate(input_ids, max_new_tokens=0)) == []


def test_stream_generate_respects_context_length():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, CONTEXT_LENGTH))
    tokens = list(runner.stream_generate(input_ids, max_new_tokens=10, do_sample=False))
    assert len(tokens) == 10


def test_stream_generate_eos_stops_early():
    runner = make_runner()
    input_ids = torch.tensor([[1, 2, 3]])
    eos_token_id = 0
    original_forward = runner.model.forward

    def fake_forward(input_ids):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, VOCAB_SIZE), -100.0)
        logits[..., eos_token_id] = 100.0
        return logits

    runner.model.forward = fake_forward
    try:
        tokens = list(
            runner.stream_generate(
                input_ids, max_new_tokens=10, do_sample=False, eos_token_id=eos_token_id
            )
        )
        assert len(tokens) == 1
    finally:
        runner.model.forward = original_forward


def test_stream_generate_text_chunks_match_generate_text():
    runner = make_runner()
    chunks = list(runner.stream_generate_text("ab", max_new_tokens=5, do_sample=False))
    assert len(chunks) == 5
    assert all(isinstance(chunk, str) for chunk in chunks)
    full = runner.generate_text("ab", max_new_tokens=5, do_sample=False)
    assert "".join(chunks) == full[len("ab"):]


def test_stream_generate_text_empty_prompt_raises():
    runner = make_runner()
    with pytest.raises(ValueError):
        list(runner.stream_generate_text("", max_new_tokens=5, do_sample=False))


def test_generate_temperature_zero_means_greedy():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    torch.manual_seed(0)
    out1 = runner.generate(input_ids, max_new_tokens=5, temperature=0.0, do_sample=True)
    torch.manual_seed(0)
    out2 = runner.generate(input_ids, max_new_tokens=5, temperature=0.0, do_sample=True)
    torch.testing.assert_close(out1, out2)
    greedy = runner.generate(input_ids, max_new_tokens=5, do_sample=False)
    torch.testing.assert_close(out1, greedy)


def test_stream_generate_temperature_zero_means_greedy():
    runner = make_runner()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    tokens = list(runner.stream_generate(input_ids, max_new_tokens=5, temperature=0.0, do_sample=True))
    assert len(tokens) == 5
    greedy = runner.generate(input_ids, max_new_tokens=5, do_sample=False)
    rebuilt = torch.cat([input_ids, *tokens], dim=-1)
    torch.testing.assert_close(rebuilt, greedy)


def test_generate_text():
    runner = make_runner()
    output = runner.generate_text("hello", max_new_tokens=5, do_sample=False)
    assert isinstance(output, str)


def test_eos_stops_generation():
    runner = make_runner()
    input_ids = torch.tensor([[1, 2, 3]])
    eos_token_id = 0
    original_forward = runner.model.forward

    def fake_forward(input_ids):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, VOCAB_SIZE), -100.0)
        logits[..., eos_token_id] = 100.0
        return logits

    runner.model.forward = fake_forward
    output = runner.generate(input_ids, max_new_tokens=10, do_sample=False, eos_token_id=eos_token_id)
    assert output.shape == (1, 4)
    runner.model.forward = original_forward


def test_from_checkpoint(tmp_path: Path):
    """Round-trip: artifacts written by the training side load into a runner."""
    artifacts = make_tiny_artifacts(tmp_path)
    runner = ModelRunner.from_checkpoint(
        checkpoint_path=artifacts["checkpoint_path"],
        tokenizer_path=artifacts["tokenizer_path"],
        model_config=artifacts["model_config"],
        device="cpu",
    )
    assert runner.model.context_length == CONTEXT_LENGTH
    assert runner.model.vocab_size == artifacts["model_config"]["vocab_size"]
    text = runner.generate_text("the cat", max_new_tokens=5, do_sample=False)
    assert isinstance(text, str)
    assert len(text) > 0


def test_from_checkpoint_legacy_model_state_dict_format(tmp_path: Path):
    artifacts = make_tiny_artifacts(tmp_path)
    checkpoint_path = tmp_path / "legacy.pt"
    model = TransformerLM(**artifacts["model_config"], device="cpu", dtype=torch.float32)
    torch.save({"model_state_dict": model.state_dict(), "step": 1}, checkpoint_path)
    runner = ModelRunner.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=artifacts["tokenizer_path"],
        model_config=artifacts["model_config"],
        device="cpu",
    )
    assert runner.model.context_length == CONTEXT_LENGTH


def test_from_checkpoint_raw_state_dict(tmp_path: Path):
    artifacts = make_tiny_artifacts(tmp_path)
    checkpoint_path = tmp_path / "raw.pt"
    model = TransformerLM(**artifacts["model_config"], device="cpu", dtype=torch.float32)
    torch.save(model.state_dict(), checkpoint_path)
    runner = ModelRunner.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=artifacts["tokenizer_path"],
        model_config=artifacts["model_config"],
        device="cpu",
    )
    assert runner.model.context_length == CONTEXT_LENGTH


def test_from_checkpoint_rejects_unsupported_format(tmp_path: Path):
    artifacts = make_tiny_artifacts(tmp_path)
    bad = tmp_path / "bad.pt"
    torch.save({"optimizer": {"fake": 1}, "step": 1}, bad)
    with pytest.raises(TypeError):
        ModelRunner.from_checkpoint(
            checkpoint_path=str(bad),
            tokenizer_path=artifacts["tokenizer_path"],
            model_config=artifacts["model_config"],
            device="cpu",
        )


def test_from_checkpoint_missing_checkpoint(tmp_path: Path):
    artifacts = make_tiny_artifacts(tmp_path)
    with pytest.raises(FileNotFoundError):
        ModelRunner.from_checkpoint(
            checkpoint_path=str(tmp_path / "missing.pt"),
            tokenizer_path=artifacts["tokenizer_path"],
            model_config=artifacts["model_config"],
            device="cpu",
        )


def test_from_checkpoint_missing_tokenizer(tmp_path: Path):
    artifacts = make_tiny_artifacts(tmp_path)
    with pytest.raises(FileNotFoundError):
        ModelRunner.from_checkpoint(
            checkpoint_path=artifacts["checkpoint_path"],
            tokenizer_path=str(tmp_path / "missing_tokenizer.json"),
            model_config=artifacts["model_config"],
            device="cpu",
        )


def make_tiny_artifacts(tmp_path: Path) -> dict:
    """Write a tokenizer.json, model_config.json and a trainer-format
    checkpoint (``{"model": ..., "step": ...}``) into ``tmp_path``.

    Mirrors what ``examples/train_tinystories.py`` + ``Trainer`` produce,
    so ``ModelRunner.from_checkpoint`` is tested against the real contract.
    """
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "the cat sat on the mat",
                "a dog ran in the park",
                "tiny stories for tiny models",
                "once upon a time there was a bird",
            ]
            * 20
        ),
        encoding="utf-8",
    )
    tokenizer = Tokenizer.train(
        files=[str(corpus)],
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": CONTEXT_LENGTH,
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "d_ff": D_FF,
        "rope_theta": ROPE_THETA,
    }
    config_path = tmp_path / "model_config.json"
    save_model_config(model_config, config_path)

    model = TransformerLM(**model_config, device="cpu", dtype=torch.float32)
    checkpoint_path = tmp_path / "latest.pt"
    torch.save({"model": model.state_dict(), "step": 42, "train_loss": 1.23}, checkpoint_path)

    return {
        "tokenizer_path": str(tokenizer_path),
        "config_path": str(config_path),
        "model_config": model_config,
        "checkpoint_path": str(checkpoint_path),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_runner_cuda():
    device = torch.device("cuda")
    runner = make_runner(device=device)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 8), device=device)
    logits = runner.forward(input_ids)
    assert logits.device.type == "cuda"
    for parameter in runner.model.parameters():
        assert parameter.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_generate_cuda():
    device = torch.device("cuda")
    runner = make_runner(device=device)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8), device=device)
    output = runner.generate(input_ids, max_new_tokens=5, do_sample=False)
    assert output.device.type == "cuda"
    assert output.shape == (1, 13)