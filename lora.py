import math
import torch.nn as nn
import torch

class LoRALayer(nn.Module):
    def __init__(self, dim_in, dim_out, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(dim_in, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, dim_out))
        self.alpha = alpha
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def replace_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearLoRA(module, rank, alpha))
        else:
            replace_with_lora(module, rank, alpha)
