import math
from typing import Optional


class LRScheduler:
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: Optional[int] = None,
        current_step: int = 0,
    ):
        if max_lr <= 0:
            raise ValueError("max_lr must be positive")
        if min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if max_lr <= min_lr:
            raise ValueError(f"max_lr ({max_lr}) must be greater than min_lr ({min_lr})")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if total_steps is not None and total_steps <= warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps")
        if current_step < 0:
            raise ValueError("current_step must be non-negative")

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = current_step
    
    
    def step(self) -> None:
        self.current_step += 1
    
    
    def get_lr(self) -> float:
        t = self.current_step
        
        if t < self.warmup_steps:
            return self.max_lr * t / max(1, self.warmup_steps)
        if self.total_steps is None:
            return self.max_lr
        if t >= self.total_steps:
            return self.min_lr
        
        progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine

    
    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }
    
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.current_step = state_dict['current_step']
        self.max_lr = state_dict['max_lr']
        self.min_lr = state_dict['min_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']