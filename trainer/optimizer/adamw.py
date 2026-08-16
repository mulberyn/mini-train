import torch
from torch.optim import Optimizer
from collections.abc import Callable
from typing import Optional


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= betas[0] < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= betas[1] < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        
        defaults = {
            'lr': lr,
            "betas": betas,
            'eps': eps,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )
                
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                m, v, step = state['m'], state['v'], state['step']
                
                step += 1
                state['step'] = step
                
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                state['m'] = m
                state['v'] = v

                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                
                p.data -= lr * weight_decay * p.data
                p.data -= lr * m_hat / (v_hat ** 0.5 + eps)
                
        return loss