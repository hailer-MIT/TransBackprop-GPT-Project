import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from TransBackprop_GPT_Project.config import GPTConfig # Assuming config.py is in the parent directory

def train_custom_bpe_tokenizer():
    config = GPTConfig()

    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Download raw data if not present
    for subset_name, url in config.RAW_DATA_URLS.items():
        file_path = os.path.join(config.DATA_DIR, f"ptb.{subset_name}.txt")
        if not os.path.exists(file_path):
            print(f"Downloading raw data from {url} to {file_path}...")
            import requests
            from tqdm import tqdm
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(file_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            print(f"Finished downloading {file_path}")

    # Paths to raw text files for training the tokenizer
    raw_data_files = [os.path.join(config.DATA_DIR, f"ptb.{subset_name}.txt") for subset_name in config.RAW_DATA_URLS.keys()]

    # Define special tokens consistently, using GPT-2's EOS token string for consistency
    gpt2_eos_token = AutoTokenizer.from_pretrained("gpt2").eos_token
    special_tokens_dict = {
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "eos_token": gpt2_eos_token,
        "cls_token": "<cls>",
        "sep_token": "<sep>",
        "mask_token": "<mask>",
    }
    special_tokens_list_for_trainer = list(special_tokens_dict.values())

    # Initialize a BPE tokenizer model
    bpe_model = BPE(unk_token=special_tokens_dict["unk_token"])
    tokenizer = Tokenizer(bpe_model)

    # Customize pre-tokenizer, normalizer
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    # Create a BpeTrainer
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=2,
        special_tokens=special_tokens_list_for_trainer
    )

    # Train the tokenizer
    print("Training custom BPE tokenizer...")
    tokenizer.train(
        files=raw_data_files,
        trainer=trainer
    )

    # Update vocab_size in config based on the trained tokenizer
    config.vocab_size = tokenizer.get_vocab_size()

    # Convert the trained `tokenizers` object to a `transformers.PreTrainedTokenizerFast`
    # Initialize without special tokens in constructor to set them via add_special_tokens later
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
    )

    # Explicitly add special tokens to the wrapped_tokenizer for proper ID mapping
    # This is crucial for `wrapped_tokenizer.eos_token_id` to be populated
    added_tokens_map = wrapped_tokenizer.add_special_tokens(special_tokens_dict)

    # Ensure the config's pad_token_id and vocab_size are updated with the custom tokenizer's values
    config.pad_token_id = wrapped_tokenizer.pad_token_id
    config.vocab_size = len(wrapped_tokenizer) # Update vocab size to reflect all added tokens
    
    # Create the directory for the custom tokenizer if it doesn't exist
    os.makedirs(config.CUSTOM_TOKENIZER_DIR, exist_ok=True)
    
    # Save the custom tokenizer
    wrapped_tokenizer.save_pretrained(config.CUSTOM_TOKENIZER_DIR)
    print(f"Custom BPE tokenizer trained and saved to {config.CUSTOM_TOKENIZER_DIR}")

if __name__ == "__main__":
    train_custom_bpe_tokenizer()
