import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
# Removed datasets import as we'll download directly
# from datasets import load_dataset
import os
import requests # For downloading files
from tqdm import tqdm # For download progress bar

from TransBackprop_GPT_Project.utils import pad_collate_fn

# URLs for raw Penn Treebank (PTB) text files
PTB_URLS = {
    'train': 'https://raw.githubusercontent.com/wojna/pure-attention/main/data/ptb.train.txt',
    'validation': 'https://raw.githubusercontent.com/wojna/pure-attention/main/data/ptb.valid.txt',
    'test': 'https://raw.githubusercontent.com/wojna/pure-attention/main/data/ptb.test.txt',
}

DATA_DIR = "./data/ptb" # Local directory to save raw data

def download_ptb_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    for split, url in PTB_URLS.items():
        filepath = os.path.join(DATA_DIR, f"ptb.{split}.txt")
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
    def __init__(self, tokenizer, block_size, split='train'):
        # Ensure dataset is downloaded
        download_ptb_dataset()
        
        filepath = os.path.join(DATA_DIR, f"ptb.{split}.txt")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize the entire text once
        self.tokenized_text = self.tokenizer(self.text, return_tensors='pt', truncation=False, padding=False)['input_ids'][0]

    def __len__(self):
        return (len(self.tokenized_text) - 1) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        chunk = self.tokenized_text[start_idx : end_idx]
        
        input_ids = chunk[:-1]
        labels = chunk[1:]

        padding_length = (self.block_size - 1) - len(input_ids)
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding_length,), -1, dtype=torch.long)])
        
        return {'input_ids': input_ids, 'labels': labels}

def get_tokenizer(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def get_loaders(tokenizer, block_size, batch_size):
    train_dataset = PTBDataset(tokenizer, block_size, split='train')
    val_dataset = PTBDataset(tokenizer, block_size, split='validation')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, tokenizer.pad_token_id))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: pad_collate_fn(batch, tokenizer.pad_token_id))

    return train_loader, val_loader
