import torch

from models.full_model import GPT
from models.loss_function import CrossEntropyLoss


class Trainer:
    "Training loop of a GPT."

    def __init__(self, model, data, seq_len, batch_size, lr=3e-4, device="cpu",
                 optimizer_cls=torch.optim.AdamW, optimizer_kwargs=None):
        self.model      = model.to(device)
        self.data       = data.to(device) # 1-D tensor of token IDs. The full training data are concatenated into one giant stream with each text separated  with an end-of-text token (<eos> := end of sequence).
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.device     = device

        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs) # ** Unpacks a dictionary into keyword arguments.

    def get_batch(self):
        "Pick a random sequence of tokens of the data with size = seq_length and return it with its targets. It is not a problem if we pick data mid text or mid sentence because each token gets well-contexted training in other windows and also learning from partial context is also useful."
        max_start = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, max_start, (self.batch_size,))
        inputs  = torch.stack([self.data[s     : s + self.seq_len    ] for s in starts])
        targets = torch.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts]) # To find the targets just shift one step right.
        return inputs, targets

    def train(self, steps):
        # In LLMs the standard approach is to train with steps. Instead of using epochs, papers measure training in tokens seen or steps (One batch processed).
        self.model.train() # Switch the model to training mode (dropout on, batch normalization mean and std.)
        for step in range(steps):
            inputs, targets = self.get_batch()

            logits = self.model(inputs)               
            loss   = CrossEntropyLoss(logits, targets)

            self.optimizer.zero_grad()                  # Clear last step’s gradients. Id we dont do this the gradients add up.
            loss.backward()                             # Backpropapgation: It computes, for every weight in the model, the gradient.
            self.optimizer.step()                       # Update the weights. 

            if step % 100 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f}")