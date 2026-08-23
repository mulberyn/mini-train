import torch
from torch import nn
from einops import rearrange
from trainer.layers.linear import Linear
from trainer.attention.attention import scaled_dot_product_attention
