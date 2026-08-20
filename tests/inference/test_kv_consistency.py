"""Cross-cache consistency: all four caches must store identical K/V and
produce identical logits for the same token stream (docs/kv_cache.md 四/五/八).
"""

import torch

from tests.inference.conftest import (
    CONTEXT_LENGTH,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    make_cache,
    make_runner,
    update_tokens,
)


def test_all_caches_store_identical_kv():
    caches = [make_cache(t, max_batch_size=2, num_blocks=64) for t in
              ["naive", "static", "dynamic", "paged"]]
    for cache in caches:
        if hasattr(cache, "allocate_sequence"):
            cache.allocate_sequence(0)
            cache.allocate_sequence(1)

    torch.manual_seed(0)
    for t in range(10):
        key = torch.randn(2, NUM_HEADS, 1, cache.head_dim)
        value = torch.randn(2, NUM_HEADS, 1, cache.head_dim)
        for cache in caches:
            update_tokens(cache, key, value, torch.tensor([t]))

    reference = caches[0]
    for cache in caches[1:]:
        for layer_idx in range(NUM_LAYERS):
            kr, vr = reference.get(layer_idx)
            kc, vc = cache.get(layer_idx)
            torch.testing.assert_close(kc, kr)
            torch.testing.assert_close(vc, vr)


def test_all_caches_produce_identical_logits():
    runner = make_runner()
    torch.manual_seed(0)
    prompt = torch.randint(0, VOCAB_SIZE, (2, 6))
    full = torch.randint(0, VOCAB_SIZE, (2, 12))

    logits_by_cache = {}
    for cache_type in ["naive", "static", "dynamic", "paged"]:
        cache = runner.build_kv_cache(
            cache_type=cache_type, max_batch_size=2, max_seq_len=CONTEXT_LENGTH,
            num_blocks=64, block_size=8,
        )
        logits_p = runner.prefill(prompt.clone(), cache)
        logits_d = runner.decode_step(full[:, 6:7], cache, positions=torch.tensor([6]))
        logits_by_cache[cache_type] = (logits_p, logits_d)

    for cache_type in ["static", "dynamic", "paged"]:
        torch.testing.assert_close(
            logits_by_cache[cache_type][0], logits_by_cache["naive"][0],
            rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            logits_by_cache[cache_type][1], logits_by_cache["naive"][1],
            rtol=1e-5, atol=1e-6,
        )


def test_dynamic_matches_naive_across_growth_boundaries():
    """Exercise the dynamic cache across capacity-doubling boundaries."""
    naive = make_cache("naive", max_batch_size=1, max_seq_len=256)
    dynamic = make_cache("dynamic", max_batch_size=1, max_seq_len=256, initial_capacity=4)

    torch.manual_seed(1)
    for t in range(70):
        key = torch.randn(1, NUM_HEADS, 1, naive.head_dim)
        value = torch.randn(1, NUM_HEADS, 1, naive.head_dim)
        update_tokens(naive, key, value, torch.tensor([t]))
        update_tokens(dynamic, key, value, torch.tensor([t]))

    for layer_idx in range(NUM_LAYERS):
        kn, vn = naive.get(layer_idx)
        kd, vd = dynamic.get(layer_idx)
        torch.testing.assert_close(kd, kn)
        torch.testing.assert_close(vd, vn)
