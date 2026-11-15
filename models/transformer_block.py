import torch.nn as nn
from selfAttention import MultiHeadAttention
from mlp import MLP

class TransformerBlock(nn.Module):
    "Transformer block."

    def __init__(self, embed_dim, num_heads, mlp_hidden=None):
        super().__init__()
        mlp_hidden = mlp_hidden or 4 * embed_dim           # The convention is to have only one hidden layer (4x) the size of the input.

        self.attn  = MultiHeadAttention(embed_dim, num_heads)
        self.mlp   = MLP([embed_dim, mlp_hidden, embed_dim], hidden_activation=nn.GELU)

        # It normalizes over the last dimension of the input.
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5) # Standarize each token to mean 0 and standard deviation 1. This keep each token's values in a stable range so gradients dont explode or vanish. It was proven empirically that it gives better results. Layer normalization has also parameters that get learned through back propagation.    
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)             

    def forward(self, x):                      # x: (batch_size, seq_len, embed_dim)
        x = x + self.attn(self.norm1(x))       # attention sublayer
        x = x + self.mlp(self.norm2(x))        # MLP sublayer
        return x                               # (batch_size, seq_len, embed_dim)