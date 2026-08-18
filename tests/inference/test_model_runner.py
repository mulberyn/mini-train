from pathlib import Path
import pytest
import torch
from inference.model_runner import ModelRunner
from trainer.model.transformer import TransformerLM
from trainer.tokenizer.bpe_tokenizer import BPETokenizer as Tokenizer


VOCAB_SIZE = 128
CONTEXT_LENGTH = 32
D_MODEL = 32
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 64
ROPE_THETA = 10000.0


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
    model = make_model()
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    assert checkpoint_path.exists()


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