import torch
import torch.nn.functional as F

def CrossEntropyLoss(logits, targets):
    "Calculate cross entropy loss function."
    # Logits: The model's guesses.
    # Targets = the correct next tokens (the truth).
    batch, seq_len, vocab_size = logits.shape
    logits  = logits.reshape(batch * seq_len, vocab_size)  # (N, vocab_size)
    targets = targets.reshape(batch * seq_len)             # (N,)
    return F.cross_entropy(logits, targets)