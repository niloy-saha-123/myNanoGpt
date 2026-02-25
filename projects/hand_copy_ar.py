import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# Hand-designed Autoregressive GPT: Copy Task
#
# Goal: given input "A B C <sep>", generate "A B C" sequentially.
# The model receives one token at a time and predicts the next.
#
# Example run:
# In: "A B C <sep>" -> predict "A"
# In: "A B C <sep> A" -> predict "B"
# In: "A B C <sep> A B" -> predict "C"
#
# We need block_size = 7 to hold "A B C <sep> A B C".
#
# The task: when we are at output position `t` (where `t` >= 4), 
# we need to copy the token from position `t - 4`. 
# For example, to predict the token after "<sep>" (which is at position 3), 
# we are at position 3, and we need to look at position -1? Wait, no.
#
# Let's trace positions:
# pos:   0   1   2   3     4   5   6
# tok:   A   B   C <sep>   A   B   C
# We want:
# at pos 3 (<sep>), predict A (from pos 0) -> look back 3
# at pos 4 (A),     predict B (from pos 1) -> look back 3
# at pos 5 (B),     predict C (from pos 2) -> look back 3
#
# So the attention pattern needs to always look back exactly 3 positions!
#
# How we do it:
# - Token embeddings: 4 dims (A, B, C, <sep>)
# - Position embeddings: 7 dims (one-hot for positions 0 to 6)
# - Total n_embd = 11
# - W_Q extracts the position `t`
# - W_K extracts position `t + 3` (so when dot-producted with Q, position `t` matches position `t - 3`)
# - W_V extracts the token embedding to pass through
# ----------------------------------------------------------------------------

vocab_size = 4   # A=0, B=1, C=2, <sep>=3
n_embd     = 11  # 4 for token content + 7 for position
n_head     = 1
n_layer    = 1
block_size = 7   # max sequence length is 7
dropout    = 0.0

device = 'cpu'

itos = {0: 'A', 1: 'B', 2: 'C', 3: '<sep>'}
stoi = {'A': 0, 'B': 1, 'C': 2, '<sep>': 3}

# ----------------------------------------------------------------------------
# same transformer architecture as v2.py
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
        
        # WE USE CAUSAL MASK HERE! (This is an autoregressive model)
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
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

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
        return self.lm_head(x)  # (B, T, vocab_size)


# ----------------------------------------------------------------------------
# set weights by hand
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
        pos_emb[i, vocab_size + i] = 1.0  # pos info starts at index 4
    model.position_embedding_table.weight.copy_(pos_emb)

    head = model.blocks[0].sa.heads[0]

    # W_Q: extracts position dims
    W_Q = torch.zeros(n_embd, n_embd)
    for t in range(block_size):
        W_Q[vocab_size + t, vocab_size + t] = 100.0
    head.query.weight.copy_(W_Q)

    # W_K: We want Q at pos `t` to match K at pos `t-3`.
    # Therefore, W_K should map input pos `t-3` to dim `vocab_size + t`.
    # This means input index `vocab_size + t - 3` maps to output index `vocab_size + t`.
    W_K = torch.zeros(n_embd, n_embd)
    for t in range(3, block_size):
        W_K[vocab_size + t, vocab_size + t - 3] = 100.0   
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
print("Testing autoregressive copy on all 27 inputs:\n")

correct = 0
total = 0
for a in range(3):
    for b in range(3):
        for c in range(3):
            input_str = ''.join(itos[i] for i in [a, b, c])
            context = [a, b, c, stoi['<sep>']]
            preds = generate_sequence(context)

            # show each token as it's generated
            for step in range(1, len(preds) + 1):
                so_far = ''.join(itos[t] for t in preds[:step])
                print(f"Input: {input_str}  Output: {so_far}")

            if preds == [a, b, c]:
                correct += 1
            total += 1
            print()

print(f"{correct}/{total} correct")
