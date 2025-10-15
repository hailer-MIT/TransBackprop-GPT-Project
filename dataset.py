import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
# Removed datasets import as we'll download directly
# from datasets import load_dataset
import os
import requests # For downloading files
from tqdm import tqdm # For download progress bar
import pickle # For caching tokenized data

from .utils import pad_collate_fn
from .config import GPTConfig # Import GPTConfig

config = GPTConfig()

# Remove hardcoded PTB_URLS and DATA_DIR
# PTB_URLS = {
#     'train': 'https://raw.githubusercontent.com/wojna/pure-attention/main/data/ptb.train.txt',
#     'validation': 'https://raw.githubusercontent.com/wojna/pure-attention/main/data/ptb.valid.txt',
#     'test': 'https://raw.githubusercontent.com/wojna/pure-attention/main/data/ptb.test.txt',
# }
# DATA_DIR = "./data/ptb" # Local directory to save raw data

def download_ptb_dataset(config):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    for split, url in config.RAW_DATA_URLS.items():
        filepath = os.path.join(config.DATA_DIR, f"ptb.{split}.txt")
        if not os.path.exists(filepath):
            print(f"Downloading {split} dataset from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for HTTP errors
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            print(f"Downloaded {split} dataset to {filepath}")
        else:
            print(f"{split} dataset already exists at {filepath}, skipping download.")

class PTBDataset(Dataset):
    def __init__(self, tokenizer, block_size, split='train', config=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split = split
        self.config = config # Store config internally

        # Ensure dataset is downloaded
        download_ptb_dataset(self.config)

        # Define cache path for tokenized data
        os.makedirs(self.config.TOKENIZED_CACHE_DIR, exist_ok=True)
        cache_filepath = os.path.join(self.config.TOKENIZED_CACHE_DIR, f"ptb_{split}_tokenized.pkl")

        if os.path.exists(cache_filepath):
            print(f"Loading tokenized data from cache: {cache_filepath}")
            with open(cache_filepath, 'rb') as f:
                self.tokenized_text = pickle.load(f)
        else:
            print(f"Tokenizing {split} dataset and saving to cache...")
            filepath = os.path.join(self.config.DATA_DIR, f"ptb.{split}.txt")
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            self.tokenized_text = self.tokenizer(text, return_tensors='pt', truncation=False, padding=False)['input_ids'][0]
            with open(cache_filepath, 'wb') as f:
                pickle.dump(self.tokenized_text, f)
            print(f"Tokenized data saved to cache: {cache_filepath}")

    def __len__(self):
        return (len(self.tokenized_text) - 1) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        chunk = self.tokenized_text[start_idx : end_idx]

        input_ids = chunk[:-1]
        labels = chunk[1:]

        # Debug prints
        # print(f"[PTBDataset] input_ids max: {input_ids.max().item()}, min: {input_ids.min().item()}")
        # print(f"[PTBDataset] labels max: {labels.max().item()}, min: {labels.min().item()}")
        # print(f"[PTBDataset] config.vocab_size: {self.config.vocab_size}")

        # Padding is handled by pad_collate_fn in DataLoader
        return {'input_ids': input_ids, 'labels': labels}

def get_tokenizer(config: GPTConfig):
    custom_tokenizer_path = config.CUSTOM_TOKENIZER_DIR
    tokenizer_cache_path = os.path.join(config.TOKENIZED_CACHE_DIR, config.tokenizer_name)

    # Try to load custom BPE tokenizer first
    if os.path.exists(custom_tokenizer_path) and os.path.isdir(custom_tokenizer_path):
        print(f"Loading custom BPE tokenizer from: {custom_tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer_path)
    # Fallback to loading/creating generic GPT-2 tokenizer if custom not found
    elif os.path.exists(tokenizer_cache_path) and os.path.isdir(tokenizer_cache_path):
        print(f"Loading generic tokenizer from cache: {tokenizer_cache_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cache_path)
    else:
        print(f"Creating and saving new generic tokenizer to cache: {tokenizer_cache_path}")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.save_pretrained(tokenizer_cache_path)

    # Ensure the config's pad_token_id and vocab_size are updated
    config.pad_token_id = tokenizer.pad_token_id
    config.vocab_size = len(tokenizer)
    
    return tokenizer

def get_loaders(tokenizer, block_size, batch_size, config):
    # Use the config.RAW_DATA_URLS dictionary directly for splits
    train_dataset = PTBDataset(tokenizer, block_size, split='train', config=config)
    val_dataset = PTBDataset(tokenizer, block_size, split='val', config=config)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, tokenizer.pad_token_id))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: pad_collate_fn(batch, tokenizer.pad_token_id))

    return train_loader, val_loader
