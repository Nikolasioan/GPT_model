import torch
from pathlib import Path

from config import settings
from cli import parse_generate_args         
from models.bpe_tokenizer import BPETokenizer
from models.full_model import GPT


@torch.no_grad()
def generate(model, ids, max_new_tokens, max_len, temperature=1.0, top_k=None):
    "Autoregressively sample new tokens, one at a time."
    model.eval()
    for _ in range(max_new_tokens):
        context = ids[:, -max_len:]
        logits  = model(context)[:, -1, :]
        logits  = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs   = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids     = torch.cat([ids, next_id], dim=1)
    return ids


args = parse_generate_args()            

data_path = Path(__file__).resolve().parent / settings.data_file
text = data_path.read_text(encoding="utf-8")
tok = BPETokenizer()
tok.train(text, num_merges=settings.num_merges)

model = GPT(
    vocab_size = len(tok.vocab),
    embed_dim  = settings.embed_dim,
    num_heads  = settings.num_heads,
    num_layers = settings.num_layers,
    max_len    = settings.max_len,
).to(settings.device)

weights_path = Path(__file__).resolve().parent / "weights" / args.weights_name
model.load_state_dict(torch.load(weights_path, map_location=settings.device))

ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=settings.device)
out = generate(model, ids, args.max_new_tokens, settings.max_len, args.temperature, args.top_k)
print(tok.decode(out[0].tolist()))