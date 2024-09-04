import torch
import torch.nn as nn

# multihead attention - we use multiple attention heads to learn different attention patterns
# this implementation is more efficient then doing the attention for each head separately
# since we can do the matrix multiplication for all the heads at once
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        # we split the output dimension by the number of heads
        self.head_dim = dim_out // num_heads
        self.W_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_o = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_size, context_size), diagonal=1))
    
    def forward(self, x):
        # the context vectors will be of dimension (num_heads * dim_out)
        b, num_tokens, _ = x.shape
        q = self.W_q(x) # matrix multiplication of batched data, now the q tensor is of shape (b, num_tokens, dim_out)
        k = self.W_k(x)
        v = self.W_v(x)

        # we permute our tensors so that it's now (batch_size, num_heads, num_tokens, head_dim)
        # that makes it more intuitive and easier to process each head independently
        # for example, we have 1 batch, 2 heads each processing 3 tokens with a head dimension of 4
        q = q.view(b, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # we want to do dot product between the queries and keys for each head
        # the matrix multiplication is carried out between the last two dimensions of the tensors and then repeated for all the heads
        attention_scores = torch.matmul(q, k.permute(0, 1, 3, 2))
        attention_scores.masked_fill_(self.mask[:num_tokens, :num_tokens] == 1, -torch.inf)

        attention_weights = torch.softmax(attention_scores / k.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # we multiply the attention weights with the values and then concatenate the heads
        context_vectors = torch.matmul(attention_weights, v).transpose(1, 2).reshape(b, num_tokens, self.dim_out)
        # we pass the concatenated heads through a linear layer to get the final context vectors
        # this is not strictly necessary, but it is common practice in LLMs
        context_vectors = self.W_o(context_vectors)

        return context_vectors
