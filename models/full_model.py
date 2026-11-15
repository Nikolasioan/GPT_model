import torch.nn as nn

from models.token_embedding import TokenEmbedding
from models.positional_encoding import PositionalEncoding
from models.transformer_block import TransformerBlock


class GPT(nn.Module):
    "Full model inference."

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len):
        super().__init__()
        self.token_embedding     = TokenEmbedding(vocab_size, embed_dim)  
        self.positional_encoding = PositionalEncoding(max_len, embed_dim) 
        self.blocks = nn.ModuleList([                                           # Container that holds a list of submodules. The important part of this is that it registers it into the model. We dont use plain python because then the optimizer would not update the parameters of the model.      
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.final_norm  = nn.LayerNorm(embed_dim)                          
        self.output_head = nn.Linear(embed_dim, vocab_size)  # Based on the output of the model, score each token by how likely it is to be the next one
 
    def forward(self, ids):                     # ids: (batch, seq_len)
        x = self.token_embedding(ids)           # (batch, seq_len, embed_dim)
        x = self.positional_encoding(x)         # (batch, seq_len, embed_dim)
        for block in self.blocks:
            x = block(x)                        # (batch, seq_len, embed_dim)
        x = self.final_norm(x)                  
        logits = self.output_head(x)            # (batch, seq_len, vocab_size)
        return logits