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
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.output_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
    
    
    def forward(
        self,
        inputs: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embedding(inputs)
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        x = self.output_norm(x)
        logits = self.lm_head(x)
        return logits