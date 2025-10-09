import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from TransBackprop_GPT_Project.config import GPTConfig
from TransBackprop_GPT_Project.model import GPT
from TransBackprop_GPT_Project.dataset import get_tokenizer, get_loaders

# --- Training and Evaluation Functions ---

def train_epoch(model, dataloader, optimizer, lr_scheduler, device, epoch, num_epochs, global_step):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits, loss = model(input_ids, targets=labels)
        
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            global_step += 1 # Increment global_step after each batch

        if (batch_idx + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Time: {time.time() - start_time:.2f}s")
            start_time = time.time()

    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 100 else float('inf')
    return avg_loss, perplexity, global_step # Return updated global_step

def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, loss = model(input_ids, targets=labels)
            if loss is not None:
                total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 100 else float('inf')
    return avg_loss, perplexity

# --- Text Generation Function ---

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=50, device='cpu'):
    model.eval()
    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)

    output_sequence = encoded_prompt

    for _ in range(max_length - encoded_prompt.size(1)):
        input_for_model = output_sequence[:, -model.config.block_size:]
        with torch.no_grad():
            logits, _ = model(input_for_model)
        
        next_token_logits = logits[:, -1, :]
        
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        else:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token)

        output_sequence = torch.cat([output_sequence, next_token], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    return generated_text

# --- Main Training Function ---

def main():
    # 1. Device Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Configuration and Tokenizer
    tokenizer = get_tokenizer()
    config = GPTConfig(
        vocab_size=len(tokenizer),
        block_size=1024,
        pad_token_id=tokenizer.pad_token_id,
        num_epochs=3
    )

    # 3. Model Initialization
    model = GPT(config).to(device)

    # 4. Data Loaders
    train_loader, val_loader = get_loaders(tokenizer, config.block_size, batch_size=4)
    
    # 5. Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_training_steps = config.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # 6. Training Loop
    best_val_loss = float('inf')
    start_epoch = 0
    global_step = 0
    
    # Try to load latest checkpoint
    checkpoint_path = "latest_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from Epoch {start_epoch}, Global Step {global_step}")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{config.num_epochs} --- (Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M)")
        train_loss, train_perplexity, global_step = train_epoch(model, train_loader, optimizer, lr_scheduler, device, epoch, config.num_epochs, global_step)
        val_loss, val_perplexity = evaluate_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1} Results: Train Loss: {train_loss:.4f} | Train Perplexity: {train_perplexity:.2f} | Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}")

        # Always save latest checkpoint after each epoch
        latest_checkpoint_path = "latest_checkpoint.pt"
        epoch_checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
        latest_checkpoint_data = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(latest_checkpoint_data, latest_checkpoint_path)
        torch.save(latest_checkpoint_data, epoch_checkpoint_path)
        print(f"Saved latest checkpoint to {latest_checkpoint_path} and epoch checkpoint to {epoch_checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "simple_gpt_best_model.pt")
            print("Saved best model checkpoint!")

    # 7. Text Generation Example (after training)
    print("\n--- Testing Text Generation ---")
    model.load_state_dict(torch.load("simple_gpt_best_model.pt"))
    model.eval()

    prompt = "Hello, how are you"
    generated_text = generate_text(model, tokenizer, prompt, device=device)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()
