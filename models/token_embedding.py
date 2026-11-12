import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    "Maps token IDs to learnable embedding vectors."

    def __init__(self, vocab_size, embed_dim): # vocab_size: Total number of tokens the tokenizer knows.
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, ids):              # Input: (batch_size, sequence_length). Batch size is if you want to put more than 1 data inputs (texts) at the same time.
        "Replace each token id with its learned embedding vector. Within this object, the same id always maps to the same vector."
        return self.embedding(ids)       # (batch_size, sequence_length, embedding_dimension)