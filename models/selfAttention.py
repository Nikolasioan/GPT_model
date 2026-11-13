import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    "Causal multi-head self-attention."

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.W_o   = nn.Linear(embed_dim, embed_dim)
        self.scale = self.d_k ** 0.5

    def forward(self, x):                          # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        H, d_k = self.num_heads, self.d_k

        Q = self.query(x)                          # (batch_size, seq_len, embed_dim)
        K = self.key(x)                            # (batch_size, seq_len, embed_dim)
        V = self.value(x)                          # (batch_size, seq_len, embed_dim)

        Q = Q.view(batch_size, seq_len, H, d_k).transpose(1, 2)   # (batch_size, H, seq_len, d_k)
        K = K.view(batch_size, seq_len, H, d_k).transpose(1, 2)   # (batch_size, H, seq_len, d_k)
        V = V.view(batch_size, seq_len, H, d_k).transpose(1, 2)   # (batch_size, H, seq_len, d_k)

        scores  = Q @ K.transpose(-2, -1) / self.scale   # (batch_size, H, seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool() # Creation of upper triangle array. Device tells us where we want the tensor to be created (CPU or GPU).
        scores = scores.masked_fill(mask, float('-inf')) # Masking: Add the (-inf) value to the tokens after the current one, so softmax turns them into 0 weight.

        weights = F.softmax(scores, dim=-1)              # (batch_size, H, seq_len, seq_len)
        out     = weights @ V                            # (batch_size, H, seq_len, d_k)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)   # (batch_size, seq_len, embed_dim)
        return self.W_o(out)                             # (batch_size, seq_len, embed_dim)