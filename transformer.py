import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward
from utils import LayerNorm

# residual connections - the input of one layer is added to the output of another layer
# this helps with the vanishing gradient problem because it keeps the gradients relatively large

# full transformer block: layer normalization -> multihead attention -> dropout -> residual connection -> layer normalization -> feedforward -> dropout -> residual connection

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            dim_in=cfg["embedding_dim"],
            dim_out=cfg["embedding_dim"],
            context_size=cfg["context_size"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["num_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.ln1 = LayerNorm(cfg["embedding_dim"])
        self.ln2 = LayerNorm(cfg["embedding_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x += residual

        return x
    
# the output of the transformer block are the context vectors for the input embeddings