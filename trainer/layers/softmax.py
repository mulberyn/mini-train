import torch
from torch import nn


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_x = torch.max(x, dim=self.dim, keepdim=True).values
        exp_x = torch.exp(x - max_x)
        sum_exp = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_exp