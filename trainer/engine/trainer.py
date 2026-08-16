import torch
from torch import nn
from typing import Iterable, Optional
from tqdm import tqdm

from dataclasses import dataclass

from trainer.utils.gradient_clip import gradient_clipping


@dataclass
class TrainStats:
    step: int
    loss: float
    grad_norm: float
    lr: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        loss_fn=None,
        device: torch.device | None = None,
        max_grad_norm: float | None = None,
        grad_clip_norm_type: float = 2.0,
        log_interval: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else nn.CrossEntropyLoss()
        )
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
                
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.grad_clip_norm_type = grad_clip_norm_type
        self.log_interval = log_interval
        self.global_step = 0
        self.model.to(self.device)
    
    
    def _move_batch(self, input_ids: torch.Tensor, labels: torch.Tensor):
        return (input_ids.to(self.device), labels.to(self.device))
    
    
    def _get_lr(self) -> float:
        if self.scheduler is not None:
            return self.scheduler.get_lr()
        return self.optimizer.param_groups[0]["lr"]
    
    
    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor):
        self.model.train()
        input_ids, labels = self._move_batch(input_ids, labels)
        
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(input_ids)
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss.backward()
        
        grad_norm = 0.0
        if self.max_grad_norm is not None:
            grad_norm = gradient_clipping(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=self.grad_clip_norm_type,
            )
        
        if self.scheduler is not None:
            self.scheduler.step()
            lr = self.scheduler.get_lr()
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        else:
            lr = self.optimizer.param_groups[0]["lr"]
            
        self.optimizer.step()
        self.global_step += 1
        return TrainStats(step=self.global_step, loss=loss.item(), grad_norm=grad_norm, lr=lr)
    
    
    def fit(self, dataloader: Iterable, num_steps: int):
        stats = []
        iterator = iter(dataloader)
        pbar = tqdm(total=num_steps, desc="Training", dynamic_ncols=True)
        for _ in range(num_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            
            input_ids, labels = batch
            train_stats = self.train_step(input_ids, labels)
            stats.append(train_stats)
            
            pbar.set_postfix({
                "loss": f"{train_stats.loss:.6f}",
                "grad": f"{train_stats.grad_norm:.6f}",
                "lr": f"{train_stats.lr:.6e}"
            })
            pbar.update(1)
        pbar.close()
        return stats