import torch
from your_module import CausalSelfAttention

# Create an instance of CausalSelfAttention
causal_self_attention = CausalSelfAttention()

# Test case 1: B = 1, T = 10, C = 16
x = torch.randn(1, 10, 16)
output = causal_self_attention.forward(x)
print(output.shape)  # Expected output: torch.Size([1, 10, 16])

# Test case 2: B = 2, T = 5, C = 32
x = torch.randn(2, 5, 32)
output = causal_self_attention.forward(x)
print(output.shape)  # Expected output: torch.Size([2, 5, 32])

# Test case 3: B = 3, T = 8, C = 64
x = torch.randn(3, 8, 64)
output = causal_self_attention.forward(x)
print(output.shape)  # Expected output: torch.Size([3, 8, 64])
