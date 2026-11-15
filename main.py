import sys
import torch
from pathlib import Path

from config import settings
from models.bpe_tokenizer import BPETokenizer
from models.full_model import GPT
from models.trainer import Trainer


data_path = Path(__file__).resolve().parent / settings.data_file
if not data_path.exists():
    sys.exit(f"Error: training file not found: {data_path}")
text = data_path.read_text(encoding="utf-8")

tok = BPETokenizer()
tok.train(text, num_merges=settings.num_merges)
ids = torch.tensor(tok.encode(text), dtype=torch.long)
print(f"device: {settings.device} | vocab: {len(tok.vocab)} | tokens: {len(ids)}")

model = GPT(
    vocab_size = len(tok.vocab),
    embed_dim  = settings.embed_dim,
    num_heads  = settings.num_heads,
    num_layers = settings.num_layers,
    max_len    = settings.max_len,
)

trainer = Trainer(
    model, ids,
    seq_len       = settings.seq_len,
    batch_size    = settings.batch_size,
    lr            = settings.lr,
    device        = settings.device,
    val_fraction  = settings.val_fraction,
    eval_interval = settings.eval_interval,
    eval_batches  = settings.eval_batches,
)
trainer.train(steps=settings.steps)

trainer.save(settings.weights_name)
print(f"saved to weights/{settings.weights_name}")