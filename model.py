import torch
import torch.nn as nn
from torch.nn import functional as F
from TransBackprop_GPT_Project.config import GPTConfig
import math

class CausalSelfAttention(nn.Module):
    """
    A multi-head self-attention module with a causal mask.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection = combines the the attention heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout) # applied to attention weights
        self.resid_dropout = nn.Dropout(config.dropout) # applied to the output before the residual connection
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Causal mask to ensure that attention is only paid to the left in the sequence
        # We store it as a buffer so it's not considered a trainable parameter
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    A simple two-layer MLP for the Transformer block.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    A single Transformer block, composed of self-attention and MLP sub-layers.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """
    The main GPT model, composed of an embedding layer, multiple Transformer blocks, and a final linear layer for token prediction.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between embedding and language model head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())/1e6:.2f} M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        # Token embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # Positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # position embeddings of shape (1, t)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)

        # Combine embeddings and apply dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final layer norm and language model head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # If targets are provided, compute cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
