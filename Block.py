from CausalSelfAttention import CausalSelfAttention
import torch.nn as nn
from LayerNorm import LayerNorm
from MLP import MLP


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # print("inside Block")
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
