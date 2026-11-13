import torch
import torch.nn.functional as F

def CrossEntropyLoss(logits, targets):
    "Calculate cross entropy loss function."
    # Logits: The model's guesses (Real numbers - The highest the number, the highest the probability to be that specific token).
    # Targets = the correct next tokens (the truth) (Integers - Represent the ID of the correct token).
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.reshape(batch_size * seq_len, vocab_size)  # (N, vocab_size)
    targets = targets.reshape(batch_size * seq_len)             # (N,)
    return F.cross_entropy(logits, targets)