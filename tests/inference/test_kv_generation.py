"""End-to-end generation tests: cache == no-cache (docs/kv_cache.md 六).

The core correctness property of any KV cache:

    Test 1: generate(tokens) without a cache
    Test 2: prefill + decode + decode + ... with a cache

must produce identical logits and identical tokens.
"""

import pytest
import torch

from tests.inference.conftest import (
    CONTEXT_LENGTH,
    VOCAB_SIZE,
    make_runner,
    update_tokens,
)

ALL_CACHE_TYPES = ["naive", "static", "dynamic", "paged"]

# Seeded *before* model construction so every run compares the same fixed
# weights. Step-wise decode and batched forward differ by ~1e-6 (matmul /
# softmax accumulation order), so greedy tokens can flip at near-ties; a
# deterministic model makes the token-level assertions stable.
MODEL_SEED = 1234


@pytest.mark.parametrize("cache_type", ALL_CACHE_TYPES)
def test_cached_generation_matches_plain_generation(cache_type):
    torch.manual_seed(MODEL_SEED)
    runner = make_runner()
    cache = runner.build_kv_cache(cache_type=cache_type, max_seq_len=CONTEXT_LENGTH)
    torch.manual_seed(0)
    prompt = torch.randint(0, VOCAB_SIZE, (1, 8))

    cached = runner.generate_with_cache(prompt.clone(), cache, max_new_tokens=6, do_sample=False)
    plain = runner.generate(prompt.clone(), max_new_tokens=6, do_sample=False)
    torch.testing.assert_close(cached, plain)


@pytest.mark.parametrize("cache_type", ALL_CACHE_TYPES)
def test_prefill_decode_logits_match_full_forward(cache_type):
    """Per-step logits from prefill+decode equal the no-cache full forward."""
    runner = make_runner()
    cache = runner.build_kv_cache(cache_type=cache_type, max_seq_len=CONTEXT_LENGTH)
    torch.manual_seed(1)
    full = torch.randint(0, VOCAB_SIZE, (1, 11))

    logits_ref = runner.forward(full)  # no cache, whole sequence at once

    logits_p = runner.prefill(full[:, :5], cache)
    torch.testing.assert_close(logits_p[:, -1], logits_ref[:, 4], rtol=1e-4, atol=1e-4)
    for i in range(6):
        pos = torch.tensor([5 + i], dtype=torch.long)
        logits_d = runner.decode_step(full[:, 5 + i:6 + i], cache, positions=pos)
        torch.testing.assert_close(logits_d[:, -1], logits_ref[:, 5 + i], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("cache_type", ALL_CACHE_TYPES)
def test_sampling_is_deterministic_and_distribution_matches(cache_type):
    """Sampled generation is reproducible, and the sampler sees the same
    distribution as the no-cache path.

    Exact sampled *tokens* cannot be asserted against the no-cache path:
    step-wise decode and batched forward differ by ~1e-6, which is enough to
    occasionally flip a multinomial draw. The contract is that the logits (and
    therefore the distribution) match within tolerance.
    """
    torch.manual_seed(MODEL_SEED)
    runner = make_runner()
    torch.manual_seed(3)
    prompt = torch.randint(0, VOCAB_SIZE, (1, 6))

    # 1. Same seed -> identical sampled output through the cache.
    cache = runner.build_kv_cache(cache_type=cache_type, max_seq_len=CONTEXT_LENGTH)
    torch.manual_seed(42)
    sampled1 = runner.generate_with_cache(
        prompt.clone(), cache, max_new_tokens=8, temperature=1.0, do_sample=True
    )
    torch.manual_seed(42)
    sampled2 = runner.generate_with_cache(
        prompt.clone(), cache, max_new_tokens=8, temperature=1.0, do_sample=True
    )
    torch.testing.assert_close(sampled1, sampled2)

    # 2. The cache path's next-token distribution matches the no-cache path.
    cache2 = runner.build_kv_cache(cache_type=cache_type, max_seq_len=CONTEXT_LENGTH)
    logits_cached = runner.prefill(prompt, cache2)[:, -1]
    logits_plain = runner.forward(prompt)[:, -1]
    torch.testing.assert_close(logits_cached, logits_plain, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        torch.softmax(logits_cached, dim=-1), torch.softmax(logits_plain, dim=-1),
        rtol=1e-3, atol=1e-5,
    )


@pytest.mark.parametrize("cache_type", ALL_CACHE_TYPES)
def test_batch_rows_share_positions(cache_type):
    """A batch of rows (same prompt length) works through the cache."""
    torch.manual_seed(MODEL_SEED)
    runner = make_runner()
    cache = runner.build_kv_cache(
        cache_type=cache_type, max_batch_size=2, max_seq_len=CONTEXT_LENGTH,
        num_blocks=64, block_size=8,
    )
    torch.manual_seed(5)
    prompts = torch.randint(0, VOCAB_SIZE, (2, 6))

    cached = runner.generate_with_cache(prompts.clone(), cache, max_new_tokens=5, do_sample=False)
    plain = runner.generate(prompts.clone(), max_new_tokens=5, do_sample=False)
    torch.testing.assert_close(cached, plain)


@pytest.mark.parametrize("cache_type", ALL_CACHE_TYPES)
def test_generate_with_cache_resets_and_reuses(cache_type):
    """A cache can be reused for a second generation after reset."""
    torch.manual_seed(MODEL_SEED)
    runner = make_runner()
    cache = runner.build_kv_cache(cache_type=cache_type, max_seq_len=CONTEXT_LENGTH)
    torch.manual_seed(7)
    prompt = torch.randint(0, VOCAB_SIZE, (1, 5))

    out1 = runner.generate_with_cache(prompt.clone(), cache, max_new_tokens=4, do_sample=False)
    out2 = runner.generate_with_cache(prompt.clone(), cache, max_new_tokens=4, do_sample=False)
    torch.testing.assert_close(out1, out2)
    # The plain path agrees with both.
    plain = runner.generate(prompt.clone(), max_new_tokens=4, do_sample=False)
    torch.testing.assert_close(out1, plain)


@pytest.mark.parametrize("cache_type", ALL_CACHE_TYPES)
def test_generate_with_cache_eos_stops(cache_type):
    runner = make_runner()
    cache = runner.build_kv_cache(cache_type=cache_type, max_seq_len=CONTEXT_LENGTH)
    eos_token_id = 0
    kv_model = runner._get_kv_model()

    def fake_kv_forward(inputs, kv_cache, positions=None):
        batch_size, seq_len = inputs.shape
        logits = torch.full((batch_size, seq_len, VOCAB_SIZE), -100.0)
        logits[..., eos_token_id] = 100.0
        return logits

    original_forward = kv_model.forward
    kv_model.forward = fake_kv_forward
    try:
        input_ids = torch.tensor([[1, 2, 3]])
        output = runner.generate_with_cache(
            input_ids, cache, max_new_tokens=10, do_sample=False, eos_token_id=eos_token_id
        )
        assert output.shape == (1, 4)
    finally:
        kv_model.forward = original_forward


@pytest.mark.parametrize("cache_type", ALL_CACHE_TYPES)
def test_kv_model_forward_shape_and_values(cache_type):
    """The cache-aware model itself returns well-shaped, finite logits."""
    runner = make_runner()
    cache = runner.build_kv_cache(cache_type=cache_type, max_seq_len=CONTEXT_LENGTH)
    kv_model = runner._get_kv_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 7))
    positions = torch.arange(7)
    runner._ensure_sequences(cache, input_ids.size(0))
    logits = kv_model(input_ids, cache, positions)
    assert logits.shape == (1, 7, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("cache_type", ALL_CACHE_TYPES)
def test_cache_beyond_context_rejected(cache_type):
    runner = make_runner()
    cache = runner.build_kv_cache(cache_type=cache_type, max_seq_len=CONTEXT_LENGTH)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, CONTEXT_LENGTH + 1))
    with pytest.raises(ValueError):
        runner.prefill(input_ids, cache)


def test_generate_with_cache_requires_2d_input():
    runner = make_runner()
    cache = runner.build_kv_cache()
    with pytest.raises(ValueError):
        runner.generate_with_cache(torch.randint(0, VOCAB_SIZE, (5,)), cache, max_new_tokens=1)


def test_build_kv_cache_unknown_type():
    runner = make_runner()
    with pytest.raises(ValueError):
        runner.build_kv_cache(cache_type="bogus")


@pytest.mark.parametrize("cache_type", ALL_CACHE_TYPES)
def test_weight_copy_is_exact(cache_type):
    """The cached model must copy the trainer model's weights exactly."""
    runner = make_runner()
    kv_model = runner._get_kv_model()
    for name, param in runner.model.state_dict().items():
        torch.testing.assert_close(kv_model.state_dict()[name], param)
