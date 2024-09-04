import torch
import torch.nn as nn
from transformer import TransformerBlock
from utils import LayerNorm

class MyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_size"], cfg["embedding_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.ln = LayerNorm(cfg["embedding_dim"])
        self.output_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        embeddings = self.embedding(input_ids)
        pos_embeddings = self.pos_embedding(torch.arange(seq_len, device=input_ids.device))
        x = embeddings + pos_embeddings
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.ln(x)
        x = self.output_head(x) # the output head will project the context vectors back to the vocabulary size and compute the logits (next token probabilities)

        return x
