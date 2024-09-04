import torch
import torch.nn as nn
from modules.gelu import GELU

# this feedforward layer is used in the transformer model
# it allows the model to learn complex patterns in the data by using multiple linear layers with a gelu activation function in between
# the first linear layer projects the input embeddings to a higher dimensional space
# the gelu activation function introduces non-linearity
# the second linear layer projects the embeddings back to the original dimension

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embedding_dim"], 4 * cfg["embedding_dim"]),
            GELU(),
            nn.Linear(4 * cfg["embedding_dim"], cfg["embedding_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)
