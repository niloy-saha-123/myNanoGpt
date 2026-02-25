import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# Hand-designed Autoregressive GPT: Reverse Task
#
# Goal: given input "A B C <sep>", generate "C B A" sequentially.
#
# Sequence:
# pos:   0   1   2   3     4   5   6
# tok:   A   B   C <sep>   C   B   A
#
# Logic:
# - At pos 3 (<sep>), output C (from pos 2).
# - At pos 4 (C), output B (from pos 1).
# - At pos 5 (B), output A (from pos 0).
#
# Notice that to generate output at pos t (for t >= 3), 
# the model needs to attend to input position `5 - t`.
# For example:
# t=3 -> attend to 5-3 = 2
# t=4 -> attend to 5-4 = 1
# t=5 -> attend to 5-5 = 0
#
# We achieve this by mapping the Q at pos `t` to match the K at pos `5-t`.
# ----------------------------------------------------------------------------

vocab_size = 4   # A=0, B=1, C=2, <sep>=3
n_embd     = 11  # 4 for token + 7 for position
n_head     = 1
n_layer    = 1
block_size = 7
dropout    = 0.0
device = 'cpu'

itos = {0: 'A', 1: 'B', 2: 'C', 3: '<sep>'}
stoi = {'A': 0, 'B': 1, 'C': 2, '<sep>': 3}

# ----------------------------------------------------------------------------
# GPT Architecture (Standard decoder-only)
# ----------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size**-0.5
        
        # WE USE CAUSAL MASK HERE!
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

# ----------------------------------------------------------------------------
# Set weights by hand
# ----------------------------------------------------------------------------

model = GPT().to(device)

with torch.no_grad():
    # token embedding: [one-hot vocab, zeros for pos]
    tok_emb = torch.zeros(vocab_size, n_embd)
    for i in range(vocab_size):
        tok_emb[i, i] = 1.0
    model.token_embedding_table.weight.copy_(tok_emb)

    # position embedding: [zeros for vocab, one-hot pos]
    pos_emb = torch.zeros(block_size, n_embd)
    for i in range(block_size):
        pos_emb[i, vocab_size + i] = 1.0
    model.position_embedding_table.weight.copy_(pos_emb)

    head = model.blocks[0].sa.heads[0]

    # W_Q: extracts position dims
    W_Q = torch.zeros(n_embd, n_embd)
    for t in range(block_size):
        W_Q[vocab_size + t, vocab_size + t] = 100.0  # scaling by 100 to harden softmax
    head.query.weight.copy_(W_Q)

    # W_K: maps input position (5-t) to query dimension t.
    # Therefore dimension `vocab_size + 5 - t` maps to `vocab_size + t`.
    W_K = torch.zeros(n_embd, n_embd)
    for t in range(3, min(block_size, 6)): # t from 3 to 5
        W_K[vocab_size + t, vocab_size + (5 - t)] = 100.0
    head.key.weight.copy_(W_K)

    # W_V: extracts token content
    W_V = torch.zeros(n_embd, n_embd)
    for i in range(vocab_size):
        W_V[i, i] = 1.0
    head.value.weight.copy_(W_V)

    # attention proj: keep only token content
    W_proj = torch.zeros(n_embd, n_embd)
    for i in range(vocab_size):
        W_proj[i, i] = 1.0
    model.blocks[0].sa.proj.weight.copy_(W_proj)
    model.blocks[0].sa.proj.bias.zero_()

    # FFN: zero out
    model.blocks[0].ffwd.net[0].weight.zero_()
    model.blocks[0].ffwd.net[0].bias.zero_()
    model.blocks[0].ffwd.net[2].weight.zero_()
    model.blocks[0].ffwd.net[2].bias.zero_()

    # LayerNorms: identity
    model.blocks[0].ln1.weight.fill_(1.0)
    model.blocks[0].ln1.bias.zero_()
    model.blocks[0].ln2.weight.fill_(1.0)
    model.blocks[0].ln2.bias.zero_()
    model.ln_f.weight.fill_(1.0)
    model.ln_f.bias.zero_()

    # unembedding: return logits based on token dimensions
    W_out = torch.zeros(vocab_size, n_embd)
    for i in range(vocab_size):
        W_out[i, i] = 1.0
    model.lm_head.weight.copy_(W_out)
    model.lm_head.bias.zero_()

# ----------------------------------------------------------------------------
# Sequence generation function
# ----------------------------------------------------------------------------

def generate_sequence(context):
    model.eval()
    idx = torch.tensor([context], device=device)
    generated_tokens = []
    for _ in range(3):
        logits = model(idx)
        next_token = logits[0, -1, :].argmax().item()
        idx = torch.cat((idx, torch.tensor([[next_token]], device=device)), dim=1)
        generated_tokens.append(next_token)
    return generated_tokens

# test on all 27 possible 3-character inputs
if __name__ == '__main__':
    print("Testing autoregressive reverse on all 27 inputs:\n")

    correct = 0
    total = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                input_str = ''.join(itos[i] for i in [a, b, c])
                context = [a, b, c, stoi['<sep>']]
                preds = generate_sequence(context)
                expected = [c, b, a]

                # show each token as it's generated
                for step in range(1, len(preds) + 1):
                    so_far = ''.join(itos[t] for t in preds[:step])
                    print(f"Input: {input_str}  Output: {so_far}")

                if preds == expected:
                    correct += 1
                total += 1
                print()

    print(f"{correct}/{total} correct")
