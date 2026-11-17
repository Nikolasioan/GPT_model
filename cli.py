import argparse

from config import settings


def parse_args():
    "Parse command-line arguments, defaulting to the config values."
    p = argparse.ArgumentParser(
        description="Train the GPT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,   # With this it also shows default values.
    )
    p.add_argument("--num_merges",    type=int,   default=settings.num_merges,    help="number of BPE merges (≈ vocab size)")
    p.add_argument("--embed_dim",     type=int,   default=settings.embed_dim,     help="embedding dimension")
    p.add_argument("--num_heads",     type=int,   default=settings.num_heads,     help="number of attention heads")
    p.add_argument("--num_layers",    type=int,   default=settings.num_layers,    help="number of transformer blocks")
    p.add_argument("--max_len",       type=int,   default=settings.max_len,       help="max sequence length (context window)")
    p.add_argument("--seq_len",       type=int,   default=settings.seq_len,       help="training window length")
    p.add_argument("--batch_size",    type=int,   default=settings.batch_size,    help="sequences per batch")
    p.add_argument("--lr",            type=float, default=settings.lr,            help="learning rate")
    p.add_argument("--steps",         type=int,   default=settings.steps,         help="number of training steps")
    p.add_argument("--val_fraction",  type=float, default=settings.val_fraction,  help="fraction of data held out for validation")
    p.add_argument("--eval_interval", type=int,   default=settings.eval_interval, help="steps between validation checks")
    p.add_argument("--eval_batches",  type=int,   default=settings.eval_batches,  help="batches used per validation")
    p.add_argument("--device",        type=str,   default=settings.device,        help="cpu or cuda")
    p.add_argument("--weights_name",  type=str,   default=settings.weights_name,  help="filename to save the weights as")
    p.add_argument("--data_file",     type=str,   default=settings.data_file,     help="path to the training text file")
    p.add_argument("--save_interval", type=int, default=settings.save_interval,   help="save a checkpoint every N steps (0 = only at the end)")
    return p.parse_args() # This reads what you typed in the terminal (Separate words by space) and turns it into an args object.

def parse_generate_args():
    "Parse command-line arguments for generation."
    p = argparse.ArgumentParser(description="Generate text with the trained GPT.")
    p.add_argument("--prompt",         type=str,   default="ROMEO:", help="starting text")
    p.add_argument("--max_new_tokens", type=int,   default=300,      help="how many tokens to generate")
    p.add_argument("--temperature",    type=float, default=0.8,      help="randomness (lower = safer)")
    p.add_argument("--top_k",          type=int,   default=50,       help="sample from the top-k tokens")
    p.add_argument("--weights_name",   type=str,   default=settings.weights_name, help="weights file to load")
    return p.parse_args()