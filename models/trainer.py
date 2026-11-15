import torch
import math

from models.full_model import GPT
from models.loss_function import CrossEntropyLoss


class Trainer:
    "Training loop of a GPT."

    def __init__(self, model, data, seq_len, batch_size, lr=3e-4, device="cpu",
                 val_fraction=0.1, eval_interval=100, eval_batches=20,
                 optimizer_cls=torch.optim.AdamW, optimizer_kwargs=None):
        self.model      = model.to(device)
        self.data       = data.to(device) 
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.device     = device
        self.eval_interval = eval_interval
        self.eval_batches  = eval_batches

        data = data.to(device) # 1-D tensor of token IDs. The full training data are concatenated into one giant stream with each text separated  with an end-of-text token (<eos> := end of sequence).
        n = int((1 - val_fraction) * len(data))
        self.train_data = data[:n] # Split data into training and valuation set.
        self.val_data   = data[n:]

        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs) # ** Unpacks a dictionary into keyword arguments.

    def get_batch(self, data):
        "Pick a random sequence of tokens of the data with size = seq_length and return it with its targets. It is not a problem if we pick data mid text or mid sentence because each token gets well-contexted training in other windows and also learning from partial context is also useful."
        max_start = len(data) - self.seq_len - 1
        starts = torch.randint(0, max_start, (self.batch_size,))
        inputs  = torch.stack([data[s     : s + self.seq_len    ] for s in starts])
        targets = torch.stack([data[s + 1 : s + self.seq_len + 1] for s in starts]) # To find the targets just shift one step right.
        return inputs, targets
    
    @torch.no_grad() # Turns off gradient tracking. Makes evaluation faster and lighter.
    def evaluate(self, k=5):
        "Evaluate with loss, perplexity, top-1 and top-k accuracy."
        self.model.eval() # Switch model to evaluation mode.
        losses, topk_accs, top1_accs = [], [], []
        for _ in range(self.eval_batches):
            inputs, targets = self.get_batch(self.val_data) 
            logits = self.model(inputs)
            losses.append(CrossEntropyLoss(logits, targets).item())  # Append the average of cross entropy across the full logits and targets sequence.
            
            topk = logits.topk(k, dim=-1).indices # top-k token ids at each position: (batch, seq_len, k).
            topk_correct = topk.eq(targets.unsqueeze(-1)).any(-1)  # True if the true token is among the k: (batch, seq_len).

            top1_correct = topk[..., 0] == targets

            top1_accs.append(top1_correct.float().mean().item())  
            topk_accs.append(topk_correct.float().mean().item()) # Append the fraction of positions where the target was in the top k.
        self.model.train() # Switch model again to train mode.
        val_loss = sum(losses) / len(losses)
        return k, val_loss, math.exp(val_loss), sum(top1_accs) / len(top1_accs), sum(topk_accs) / len(topk_accs)  # Returns loss, perplexity and top-k accuracy, top-1 accuracy.


    def train(self, steps):
        # In LLMs the standard approach is to train with steps. Instead of using epochs, papers measure training in tokens seen or steps (One batch processed).
        self.model.train() # Switch the model to training mode (dropout on, batch normalization mean and std.)
        for step in range(steps):
            inputs, targets = self.get_batch(self.train_data)

            logits = self.model(inputs)               
            loss   = CrossEntropyLoss(logits, targets)

            self.optimizer.zero_grad()                  # Clear last step’s gradients. Id we dont do this the gradients add up.
            loss.backward()                             # Backpropapgation: It computes, for every weight in the model, the gradient.
            self.optimizer.step()                       # Update the weights. 

            if step % self.eval_interval == 0:
                            k, val_loss, val_ppl, val_top1, val_topk = self.evaluate()
                            print(f"step {step:5d} | train loss {loss.item():.4f} "
                                f"| val loss {val_loss:.4f} | val ppl {val_ppl:6.2f} | val top-1 acc{val_top1:.3f} | val top-{k} acc {val_topk:.3f}")