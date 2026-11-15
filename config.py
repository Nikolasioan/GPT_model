import torch
from configuration.base_config import BaseConfig


class Settings(BaseConfig):
    "Default configuration values."

    # tokenizer
    num_merges = 500

    # model
    embed_dim  = 128
    num_heads  = 4
    num_layers = 4
    max_len    = 256

    # training
    seq_len    = 128
    batch_size = 16
    lr         = 3e-4
    steps      = 2000

    # evaluation
    val_fraction  = 0.1
    eval_interval = 100
    eval_batches  = 20

    # settings
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    weights_name = "gpt.pt"
    data_file    = "data/input.txt"


settings = Settings()