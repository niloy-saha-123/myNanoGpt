"""
Shared, parameterized GPT architecture for the research experiments.

This is the same decoder-only transformer used in all hand-designed models
(and in v2.py), but with hyperparameters passed as constructor arguments
instead of module-level globals.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Simple linear → ReLU → linear → dropout."""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Tiny decoder-only GPT.

    Args:
        vocab_size: number of tokens
        n_embd:     embedding dimension
        n_head:     number of attention heads
        n_layer:    number of transformer blocks
        block_size: maximum sequence length
        dropout:    dropout rate (default 0.0)
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def get_attention_weights(self, idx):
        """
        Run a forward pass and return the attention weight matrices
        from every head in every layer.

        Returns:
            list of list of tensors: weights[layer][head] is (B, T, T)
        """
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        all_weights = []
        for block in self.blocks:
            x_ln = block.ln1(x)
            layer_weights = []
            head_outputs = []
            for head in block.sa.heads:
                k = head.key(x_ln)
                q = head.query(x_ln)
                wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
                wei = wei.masked_fill(head.tril[:T, :T] == 0, float('-inf'))
                wei = F.softmax(wei, dim=-1)
                layer_weights.append(wei.detach())
                v = head.value(x_ln)
                head_outputs.append(wei @ v)
            all_weights.append(layer_weights)
            attn_out = torch.cat(head_outputs, dim=-1)
            attn_out = block.sa.dropout(block.sa.proj(attn_out))
            x = x + attn_out
            x = x + block.ffwd(block.ln2(x))

        return all_weights
