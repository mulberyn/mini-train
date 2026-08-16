import torch
from torch import nn

from trainer.layers import Embedding, RMSNorm, Linear
from trainer.model import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        self.token_embedding = Embedding(
            vocab_size, 
            d_model, 
            device=device, 
            dtype=dtype
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,  # 传递给 RoPE
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.output_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )
    
    
    def forward(
        self,
        inputs: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(
                f"inputs must have shape [B, S], "
                f"got {inputs.shape}"
            )
        if inputs.dtype != torch.long:
            raise TypeError(
                f"inputs must be torch.long, "
                f"got {inputs.dtype}"
            )
        if inputs.size(1) > self.context_length:
            raise ValueError(
                f"sequence length {inputs.size(1)} exceeds "
                f"context length {self.context_length}"
            )

        x = self.token_embedding(inputs)
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        x = self.output_norm(x)
        logits = self.lm_head(x)
        return logits