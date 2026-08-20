from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator
import torch
from trainer.model.transformer import TransformerLM
from trainer.tokenizer.bpe_tokenizer import BPETokenizer as Tokenizer

from inference.kv_cache.base import KVCache
from inference.kv_cache.dynamic import DynamicKVCache
from inference.kv_cache.naive import NaiveKVCache
from inference.kv_cache.paged import PagedKVCache
from inference.kv_cache.static import StaticKVCache
from inference.kv_model import KVCachedTransformerLM


class ModelRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        eos_token_id: int | None = None,
    ):
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(self.device)
        if dtype is not None:
            self.model.to(dtype=dtype)
        self.model.eval()
        self.eos_token_id = eos_token_id
        self._kv_model: KVCachedTransformerLM | None = None


    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        model_config: dict[str, Any],
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        eos_token_id: int | None = None,
    ) -> "ModelRunner":
        checkpoint_path = Path(checkpoint_path)
        tokenizer_path = Path(tokenizer_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        tokenizer = Tokenizer.load(tokenizer_path)
        model = TransformerLM(**model_config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = cls._extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            eos_token_id=eos_token_id,
        )


    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        """Extract a model ``state_dict`` from any supported checkpoint format.

        Supported formats:

        - ``{"model": <state_dict>, ...}`` (produced by the Trainer)
        - ``{"model_state_dict": <state_dict>, ...}`` (legacy)
        - a bare ``state_dict`` (``torch.save(model.state_dict(), ...)``)
        """
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                return checkpoint["model"]
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                return checkpoint
        raise TypeError(
            "Unsupported checkpoint format. Expected a bare state_dict, or "
            "a dictionary containing 'model' or 'model_state_dict'."
        )


    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)


    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self.tokenizer.decode(token_ids)


    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        return self.model(input_ids)


    @staticmethod
    def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        return logits / temperature


    @staticmethod
    def _top_k(logits: torch.Tensor, k: int | None) -> torch.Tensor:
        if k is None:
            return logits
        if k <= 0:
            raise ValueError(f"top_k must be positive, got {k}")
        vocab_size = logits.size(-1)
        if k > vocab_size:
            k = vocab_size
        values, _ = torch.topk(logits, k, dim=-1)
        threshold = values[..., -1, None]
        return torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )


    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        if not do_sample:
            return torch.argmax(logits, dim=-1)
        logits = ModelRunner._apply_temperature(logits, temperature)
        logits = ModelRunner._top_k(logits, top_k)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        generated = input_ids
        for next_token in self.stream_generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
        ):
            generated = torch.cat([generated, next_token], dim=-1)
        return generated


    @torch.inference_mode()
    def stream_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
    ) -> Iterator[torch.Tensor]:
        """Generate tokens one at a time, yielding each new token immediately.

        Like :meth:`generate`, but instead of returning the full sequence it
        yields every newly sampled token (shape ``[B, 1]``) as soon as it is
        produced, so callers can stream output (e.g. print tokens as they
        arrive) without waiting for the whole sequence. Stops early when every
        sequence in the batch hits ``eos_token_id``.

        Example:
            >>> for token in runner.stream_generate(input_ids, max_new_tokens=64):
            ...     print(token.item(), flush=True)
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        generated = input_ids
        eos_token_id = self.eos_token_id if eos_token_id is None else eos_token_id
        context_length = self.model.context_length
        # temperature <= 0 means greedy decoding, even when do_sample=True.
        do_sample = do_sample and temperature > 0
        for _ in range(max_new_tokens):
            if generated.size(1) > context_length:
                model_input = generated[:, -context_length:]
            else:
                model_input = generated
            logits = self.model(model_input)
            next_token_logits = logits[:, -1, :]
            next_token = self._sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
            )
            next_token = next_token.unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)
            yield next_token
            if eos_token_id is not None:
                finished = (next_token.squeeze(-1) == eos_token_id)
                if torch.all(finished):
                    break


    @torch.inference_mode()
    def stream_generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
    ) -> Iterator[str]:
        """Generate text from a prompt, yielding decoded chunks incrementally.

        Each yielded string is the newly decoded portion of the *generated*
        text (the prompt itself is never yielded); join the chunks to
        reconstruct the full generated text. Intended for streaming output:

        Example:
            >>> for chunk in runner.stream_generate_text("Once upon a time", max_new_tokens=64):
            ...     print(chunk, end="", flush=True)

        Note: decoding is prefix-stable for regular text. A multi-byte UTF-8
        character split across token boundaries may briefly show up as a
        replacement character in a single chunk before the full character is
        decoded.
        """
        token_ids = self.encode(prompt)
        if not token_ids:
            raise ValueError("Prompt produces zero tokens.")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        generated_ids: list[int] = []
        prev_text = ""
        for next_token in self.stream_generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
        ):
            generated_ids.append(int(next_token))
            text = self.tokenizer.decode(generated_ids)
            yield text[len(prev_text):]
            prev_text = text


    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
    ) -> str:
        token_ids = self.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        generated_ids = self.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
        )
        return self.decode(generated_ids[0])

    # ------------------------------------------------------------------ #
    # KV-cache based inference (prefill / decode / generate_with_cache)
    # ------------------------------------------------------------------ #
    def _get_kv_model(self) -> KVCachedTransformerLM:
        """Return (and lazily build) the cache-aware twin of ``self.model``.

        The weights are copied from the runner's model, so the two produce
        identical logits -- which is exactly what the correctness tests assert.
        The random init during construction is discarded by ``load_state_dict``,
        so the RNG state is snapshotted and restored to keep the construction
        side-effect-free (otherwise the first call would shift downstream RNG
        draws, e.g. sampling).
        """
        if self._kv_model is None:
            rope_theta = self._extract_rope_theta()
            rng_state = torch.get_rng_state()
            try:
                self._kv_model = KVCachedTransformerLM(
                    vocab_size=self.model.vocab_size,
                    context_length=self.model.context_length,
                    d_model=self.model.d_model,
                    num_layers=self.model.num_layers,
                    num_heads=self.model.num_heads,
                    d_ff=self.model.d_ff,
                    rope_theta=rope_theta,
                    num_kv_heads=None,  # MHA for now
                    device=self.device,
                    dtype=next(self.model.parameters()).dtype,
                )
                state_dict = self.model.state_dict()
                self._kv_model.load_state_dict(state_dict)
                self._kv_model.to(self.device)
                self._kv_model.eval()
            finally:
                torch.set_rng_state(rng_state)
        return self._kv_model

    def _extract_rope_theta(self) -> float:
        """Recover ``rope_theta`` from the model's first RoPE buffer."""
        inv_freq = self.model.transformer_blocks[0].mha.pos_enc.inv_freq
        d_k = inv_freq.numel() * 2
        # inv_freq[i] = theta^(-2i/d_k)  ->  theta = inv_freq[1]^(-d_k/2)
        return float(inv_freq[1].pow(-d_k / 2).item())

    def build_kv_cache(
        self,
        cache_type: str = "naive",
        max_batch_size: int | None = None,
        max_seq_len: int | None = None,
        num_blocks: int | None = None,
        block_size: int = 16,
        **kwargs: Any,
    ) -> KVCache:
        """Build a KV cache configured for this runner's model.

        Args:
            cache_type: ``"naive"``, ``"static"``, ``"dynamic"`` or ``"paged"``.
            max_batch_size: default ``1``.
            max_seq_len: default ``model.context_length``.
            num_blocks: (paged) physical blocks; default enough for one full
                ``max_seq_len`` sequence per batch row.
            block_size: (paged) tokens per block.

        Returns:
            A fresh :class:`KVCache` ready for :meth:`prefill`.
        """
        if max_batch_size is None:
            max_batch_size = 1
        if max_seq_len is None:
            max_seq_len = self.model.context_length
        if num_blocks is None:
            num_blocks = max(1, (max_seq_len + block_size - 1) // block_size) * max_batch_size

        dtype = next(self.model.parameters()).dtype
        num_kv_heads = self.model.num_heads
        head_dim = self.model.d_model // self.model.num_heads
        common = dict(
            num_layers=self.model.num_layers,
            max_batch_size=max_batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=self.device,
        )
        if cache_type == "naive":
            return NaiveKVCache(max_seq_len=max_seq_len, **common)
        if cache_type == "static":
            return StaticKVCache(max_seq_len=max_seq_len, **common)
        if cache_type == "dynamic":
            return DynamicKVCache(
                max_seq_len=max_seq_len,
                initial_capacity=kwargs.pop("initial_capacity", 64),
                growth_factor=kwargs.pop("growth_factor", 2.0),
                **common,
            )
        if cache_type == "paged":
            return PagedKVCache(
                num_blocks=num_blocks, block_size=block_size, **common
            )
        raise ValueError(
            f"unknown cache_type {cache_type!r} (expected naive/static/dynamic/paged)"
        )

    @staticmethod
    def _ensure_sequences(kv_cache: KVCache, batch: int) -> None:
        """Allocate paged-cache sequence slots for batch rows ``0..batch-1``."""
        if hasattr(kv_cache, "allocate_sequence"):
            for row in range(batch):
                if row not in kv_cache.sequences:
                    kv_cache.allocate_sequence(row)

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """Process the prompt chunk, storing its K/V into ``kv_cache``.

        Args:
            input_ids: ``[B, T]`` prompt tokens (all rows share the same
                length and start at absolute position 0).
            kv_cache: cache to append to (must be empty or already holding the
                prefix that precedes ``input_ids``).

        Returns:
            ``[B, T, vocab_size]`` logits for the prompt tokens.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        self._ensure_sequences(kv_cache, input_ids.size(0))
        positions = torch.arange(input_ids.size(1), device=self.device, dtype=torch.long)
        kv_model = self._get_kv_model()
        return kv_model(input_ids, kv_cache, positions)

    @torch.inference_mode()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode one token per row, attending over the cached prefix.

        Note: named ``decode_step`` (not ``decode``) because :meth:`decode`
        already decodes token ids back to text.

        Args:
            input_ids: ``[B, 1]`` next token for every row.
            kv_cache: cache already holding the prefix (e.g. from :meth:`prefill`).
            positions: absolute positions ``[1]`` of the new token, shared by
                all rows; defaults to the current cached length.

        Returns:
            ``[B, 1, vocab_size]`` logits for the new token.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, 1], got {tuple(input_ids.shape)}")
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        if positions is None:
            # Default: append right after the current cached prefix.
            length = kv_cache.get(0)[0].shape[-2]
            positions = torch.tensor([length], device=self.device, dtype=torch.long)
        kv_model = self._get_kv_model()
        return kv_model(input_ids, kv_cache, positions)

    @torch.inference_mode()
    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation on top of a KV cache.

        Runs :meth:`prefill` for the prompt, samples the first new token from
        the prompt logits, then runs :meth:`decode_step` for every following
        token, appending each one to the cache. The returned tensor contains
        the prompt followed by the generated tokens (``[B, T_prompt + new]``).

        The cache is reset at the start of the call.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        kv_cache.reset()
        eos_token_id = self.eos_token_id if eos_token_id is None else eos_token_id
        do_sample = do_sample and temperature > 0

        generated = input_ids
        if max_new_tokens == 0:
            return generated

        # The first new token is sampled from the prompt's last logits; its
        # K/V is *not* computed here -- prefill already stored the prompt.
        logits = self.prefill(input_ids, kv_cache)
        next_token = self._sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
        ).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)
        if eos_token_id is not None and bool((next_token.squeeze(-1) == eos_token_id).all()):
            return generated

        # Every following token is fed to decode_step, which appends its K/V
        # at its absolute position before attending over the cache.
        current_pos = input_ids.size(1)
        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(
                generated[:, -1:], kv_cache,
                positions=torch.tensor([current_pos], device=self.device, dtype=torch.long),
            )
            next_token = self._sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
            ).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)
            current_pos += 1
            if eos_token_id is not None:
                finished = next_token.squeeze(-1) == eos_token_id
                if torch.all(finished):
                    break
        return generated