from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator
import torch
from trainer.model.transformer import TransformerLM
from trainer.tokenizer.bpe_tokenizer import BPETokenizer as Tokenizer


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