import torch
from typing import Callable, Protocol


class Scheduler(Protocol):
    def step(self) -> None:
        ...

    def get_lr(self) -> float:
        ...


class TrainStep:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Scheduler | None = None,
        loss_fn: Callable | None = None,
        gradient_clip_norm: float | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else torch.nn.CrossEntropyLoss()
        )
        self.gradient_clip_norm = gradient_clip_norm
    
    
    def step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        logits = self.model(input_ids)
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss.backward()
        
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item()