import dataclasses

@dataclasses.dataclass
class GPTConfig:
    vocab_size: int = 50257  # GPT-2 vocab size + 1 for padding
    block_size: int = 1024  # Max sequence length
    n_layer: int = 2  # Reduced for faster CPU training
    n_head: int = 2  # Reduced for faster CPU training
    n_embd: int = 128  # Reduced for faster CPU training
    dropout: float = 0.1  # Dropout rate
    bias: bool = True  # Use bias in linear layers
    pad_token_id: int = -1 # Will be set by tokenizer, -1 for ignore_index default
    num_epochs: int = 3 # Number of training epochs
