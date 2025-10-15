from dataclasses import dataclass
import os

@dataclass
class GPTConfig:
    # Model hyperparameters
    vocab_size: int = 50257            # GPT-2 vocab size + 1 for pad token
    block_size: int = 256              # Maximum sequence length
    n_layer: int = 6                   # Number of transformer layers
    n_head: int = 8                    # Number of attention heads
    n_embd: int = 256                  # Embedding dimension
    dropout: float = 0.1               # Dropout rate
    bias: bool = True                  # Use bias in linear layers

    # Training parameters
    batch_size: int = 8              # Batch size for training
    num_epochs: int = 20             # Number of training epochs
    learning_rate: float = 3e-4      # Learning rate
    weight_decay: float = 0.1        # Weight decay for AdamW
    clip_grad_norm: float = 1.0      # Gradient clipping

    # Data parameters
    pad_token_id: int = 50256       # Token ID for padding (vocab_size - 1)
    tokenizer_name: str = "gpt2"    # Tokenizer to use
    max_length: int = block_size    # Max length for tokenization

    # Paths for local data management
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    TOKENIZED_CACHE_DIR: str = os.path.join(DATA_DIR, "tokenized_cache")
    CUSTOM_TOKENIZER_DIR: str = os.path.join(BASE_DIR, "custom_tokenizer_output")

    # Raw data URLs for PTB dataset
    RAW_DATA_URLS = {
        'train': "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt",
        'val': "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt",
        'test': "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt",
    }
