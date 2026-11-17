import torch
from pathlib import Path

from config import settings
from cli import parse_generate_args         
from models.bpe_tokenizer import BPETokenizer
from models.full_model import GPT


@torch.no_grad() # Turn off gradient tracking.
def generate(model, ids, max_new_tokens, max_len, temperature=1.0, top_k=None):
    "Autoregressively sample new tokens, one at a time."
    model.eval() # In evaluation mode dropout is disabled.
    for _ in range(max_new_tokens): # max_new_tokens is how much tokens to generate at a time.
        context = ids[:, -max_len:] # Attend only to the max_len last tokens. If the text grows past that, cut it.
        logits  = model(context)[:, -1, :] # The model returns logits for every position. We only care about the prediction for the next token, which comes from the last position.
        logits  = logits / temperature # Deviding the logits before softmax reshapes the distribution. Temperature > 1 (model more random), Temperature = 1 (leaves it unchanged) and Temperature < 1 (model less random).
        if top_k is not None: # Restricts sampling to only the top_k most likely tokens.
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf") # Any logit below that threshold (top_k) is set to -inf, so after softmax it becomes probability 0
        probs   = torch.softmax(logits, dim=-1) # Calculate the probabilities.
        next_id = torch.multinomial(probs, num_samples=1) # Sample one token id from that distribution (Randomness). Multinomial is weighted random sampling (according to the props).
        ids     = torch.cat([ids, next_id], dim=1) # Put the new token at the end.
    return ids


args = parse_generate_args()            

data_path = Path(__file__).resolve().parent / settings.data_file
text = data_path.read_text(encoding="utf-8")
tok = BPETokenizer()
tok.train(text, num_merges=settings.num_merges) # You have to use the exact same encoding that you used on training.

# You have to be sure that the settings you use are the ones the model was trained for.
model = GPT(
    vocab_size = len(tok.vocab),
    embed_dim  = settings.embed_dim,
    num_heads  = settings.num_heads,
    num_layers = settings.num_layers,
    max_len    = settings.max_len,
).to(settings.device)

weights_path = Path(__file__).resolve().parent / "weights" / args.weights_name
model.load_state_dict(torch.load(weights_path, map_location=settings.device)) # Load the weights to the model.

ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=settings.device) # Encode the input prompt.
out = generate(model, ids, args.max_new_tokens, settings.max_len, args.temperature, args.top_k)
print(tok.decode(out[0].tolist()))