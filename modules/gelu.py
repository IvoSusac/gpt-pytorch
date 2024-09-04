import torch
import torch.nn as nn

# gelu activation function - a smooth approximation of the relu activation function
# computationally efficient approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# advantages over relu: differentiable everywhere, non-zero gradients for negative values so less neurons are inactive during training
class GELU (nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)))

