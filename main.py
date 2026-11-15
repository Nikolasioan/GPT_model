import sys
import torch
from pathlib import Path

from cli import parse_args
from models.bpe_tokenizer import BPETokenizer
from models.full_model import GPT
from models.trainer import Trainer


args = parse_args()

data_path = Path(__file__).resolve().parent / args.data_file
if not data_path.exists():
    sys.exit(f"Error: training file not found: {data_path}")
text = data_path.read_text(encoding="utf-8")

tok = BPETokenizer()
tok.train(text, num_merges=args.num_merges)
ids = torch.tensor(tok.encode(text), dtype=torch.long)
print(f"device: {args.device} | vocab: {len(tok.vocab)} | tokens: {len(ids)}")

model = GPT(
    vocab_size = len(tok.vocab),
    embed_dim  = args.embed_dim,
    num_heads  = args.num_heads,
    num_layers = args.num_layers,
    max_len    = args.max_len,
)

trainer = Trainer(
    model, ids,
    seq_len       = args.seq_len,
    batch_size    = args.batch_size,
    lr            = args.lr,
    device        = args.device,
    val_fraction  = args.val_fraction,
    eval_interval = args.eval_interval,
    eval_batches  = args.eval_batches,
)
trainer.train(steps=args.steps)

trainer.save(args.weights_name)
print(f"saved to weights/{args.weights_name}")