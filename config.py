import torch
from configuration.base_config import BaseConfig


class Settings(BaseConfig):
    "Default configuration values."

    # tokenizer
    num_merges = 2000

    # model
    embed_dim  = 384 
    num_heads  = 6
    num_layers = 6
    max_len    = 256

    # training
    seq_len    = 256
    batch_size = 32
    lr         = 3e-4
    steps = 20000
    save_interval = 2000 
    
    # evaluation
    val_fraction  = 0.1
    eval_interval = 250
    eval_batches  = 20

    # settings
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    weights_name = "gpt.pt"
    data_file    = "data/input.txt"


settings = Settings()