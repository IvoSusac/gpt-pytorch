import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        # the model can learn to scale or shift the normalized values by using the gamma and beta parameters if it needs to (if it improves the performance)
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps # small constant added to the variance to prevent division by zero

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # we use the biased variance (dividing by n instead of n - 1) because it's compatible to the pretrained weights we will load
        # for large embedding dimensions, the difference between n and n - 1 is negligible
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta

