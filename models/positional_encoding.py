import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    "Adds a learned position vector to each token's embedding. Positional encoding fixes the problem that attention is order-blind."

    def __init__(self, max_length, embed_dim): # Max length is the longest sequence of tokens the model can handle as text input.
        super().__init__()
        self.pos_embedding = nn.Embedding(max_length, embed_dim)  

    def forward(self, x): # Input: (batch_size, sequence_length). Batch size is if you want to put more than 1 data inputs (texts) at the same time.
        sequence_length = x.size(1) 
        positions = torch.arange(sequence_length, device=x.device)  # Build a list from zero to (sequence_length - 1)
        return x + self.pos_embedding(positions) # Add position info to each token. The input x is the token embedding.