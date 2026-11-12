import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    "Self attention model."
    
    def __init__(self, embed_dim, dimensions_k, dimensions_v):
        super().__init__()
        self.query = nn.Linear(embed_dim, dimensions_k)
        self.key   = nn.Linear(embed_dim, dimensions_k)
        self.value = nn.Linear(embed_dim, dimensions_v)
        self.scale = dimensions_k ** 0.5          

    def forward(self, x):                          # x: (batch, seq_len, embed_dim)
        Q = self.query(x)                          # (b, seq_len, dimensions_k):
        K = self.key(x)                            # (b, seq_len, dimensions_k)
        V = self.value(x)                          # (b, seq_len, dimensions_v)

        scores  = Q @ K.transpose(-2, -1) / self.scale   # (b, seq_len, seq_len)
        weights = F.softmax(scores, dim=-1)              # (b, seq_len, seq_len)
        out     = weights @ V                            # (b, seq_len, dimensions_v)
        return out
    