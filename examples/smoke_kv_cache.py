"""Smoke test: KV-cache generation must match no-cache generation.

Runs prefill + decode with every KV cache implementation and verifies that
(a) the generated tokens equal the no-cache path and (b) the per-step logits
match the full-sequence forward within tolerance.

Run:
    python -m examples.smoke_kv_cache [--device cpu|cuda]
"""

from __future__ import annotations

import argparse

import torch

from inference.model_runner import ModelRunner
from trainer.model.transformer import TransformerLM

VOCAB_SIZE, CONTEXT_LENGTH = 128, 32
D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF = 32, 2, 4, 64
CACHE_TYPES = ["naive", "static", "dynamic", "paged"]


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % VOCAB_SIZE for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(int(i)) for i in ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, d_model=D_MODEL,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, d_ff=D_FF,
        rope_theta=10000.0, device=device, dtype=torch.float32,
    )
    runner = ModelRunner(model=model, tokenizer=DummyTokenizer(), device=device)

    prompt = torch.randint(0, VOCAB_SIZE, (1, 8), device=device)
    for cache_type in CACHE_TYPES:
        cache = runner.build_kv_cache(
            cache_type=cache_type, max_batch_size=1, max_seq_len=CONTEXT_LENGTH,
        )
        with_cache = runner.generate_with_cache(
            prompt.clone(), cache, max_new_tokens=6, do_sample=False
        )
        no_cache = runner.generate(prompt.clone(), max_new_tokens=6, do_sample=False)
        assert torch.equal(with_cache, no_cache), (cache_type, with_cache, no_cache)
        print(f"{cache_type:8s} cache == no-cache tokens: OK")

    # Per-step logits: prefill + decode must equal the full forward.
    full = torch.randint(0, VOCAB_SIZE, (1, 11), device=device)
    logits_ref = runner.forward(full)
    for cache_type in CACHE_TYPES:
        cache = runner.build_kv_cache(
            cache_type=cache_type, max_batch_size=1, max_seq_len=CONTEXT_LENGTH,
        )
        logits_p = runner.prefill(full[:, :5], cache)
        torch.testing.assert_close(logits_p[:, -1], logits_ref[:, 4], rtol=1e-4, atol=1e-4)
        for i in range(6):
            pos = torch.tensor([5 + i], device=device, dtype=torch.long)
            logits_d = runner.decode_step(full[:, 5 + i:6 + i], cache, positions=pos)
            torch.testing.assert_close(
                logits_d[:, -1], logits_ref[:, 5 + i], rtol=1e-4, atol=1e-4
            )
        print(f"{cache_type:8s} prefill/decode logits == full forward: OK")

    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
