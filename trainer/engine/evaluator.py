from __future__ import annotations
import math
import torch


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        device: torch.device,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device


    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        num_batches: int,
    ) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for step, (input_ids, labels) in enumerate(dataloader):
            if step >= num_batches:
                break
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(input_ids)
            loss = self.loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            num_tokens = labels.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        if total_tokens == 0:
            raise RuntimeError("No validation batches were evaluated.")
        loss = total_loss / total_tokens
        ppl = math.exp(loss) if loss < 20 else float("inf")
        return {
            "loss": loss,
            "ppl": ppl,
        }