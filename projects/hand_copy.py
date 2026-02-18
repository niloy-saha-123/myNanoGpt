import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# Hand-designed GPT: Copy Task
#
# The goal here is to manually set the weights of a tiny transformer so that
# it copies its input. No training at all -- we figure out the right numbers
# by hand and just plug them in.
#
# Vocab: A=0, B=1, C=2 (3 tokens)
# Input:  A B C  (any 3-character sequence)
# Output: A B C  (exact same sequence)
#
# How it works:
#   - Token embedding is the identity matrix, so A -> [1,0,0], B -> [0,1,0], C -> [0,0,1]
#   - Position embedding is all zeros (position doesn't matter for copying)
#   - Attention does nothing (weights zeroed out, residual passes embedding through)
#   - FFN does nothing (weights zeroed out)
#   - Unembedding (lm_head) is also identity, so [1,0,0] -> logit for A is highest
#
# The residual stream just carries the token embedding all the way through unchanged.
# ----------------------------------------------------------------------------

# tiny config -- just enough to do the job
vocab_size = 3   # A, B, C
n_embd     = 3   # one dimension per token (identity embedding)
n_head     = 1
n_layer    = 1
block_size = 3   # sequences of length 3
dropout    = 0.0

device = 'cpu'  # not using GPU here, the model is tiny

# token names for printing
itos = {0: 'A', 1: 'B', 2: 'C'}
stoi = {'A': 0, 'B': 1, 'C': 2}


# ----------------------------------------------------------------------------
# same transformer architecture as v2.py, just parameterized differently
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
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
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
        tok_emb = self.token_embedding_table(idx)                              # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab_size)


# ----------------------------------------------------------------------------
# build the model and set weights by hand
# ----------------------------------------------------------------------------

model = GPT().to(device)

with torch.no_grad():
    # token embedding = identity
    # A -> [1,0,0], B -> [0,1,0], C -> [0,0,1]
    model.token_embedding_table.weight.copy_(torch.eye(vocab_size))

    # position embedding = zeros (position doesn't matter for copying)
    model.position_embedding_table.weight.zero_()

    # attention: zero out Q, K, V, and projection so attention contributes nothing
    # the residual connection will carry the embedding through unchanged
    for block in model.blocks:
        for head in block.sa.heads:
            head.key.weight.zero_()
            head.query.weight.zero_()
            head.value.weight.zero_()
        block.sa.proj.weight.zero_()
        block.sa.proj.bias.zero_()

        # FFN: zero out so it contributes nothing either
        block.ffwd.net[0].weight.zero_()
        block.ffwd.net[0].bias.zero_()
        block.ffwd.net[2].weight.zero_()
        block.ffwd.net[2].bias.zero_()

        # LayerNorm: set to identity (weight=1, bias=0 is the default, but be explicit)
        block.ln1.weight.fill_(1.0)
        block.ln1.bias.zero_()
        block.ln2.weight.fill_(1.0)
        block.ln2.bias.zero_()

    # final LayerNorm: identity
    model.ln_f.weight.fill_(1.0)
    model.ln_f.bias.zero_()

    # unembedding = identity (transpose of token embedding)
    # so [1,0,0] -> logit for A is 1, logit for B is 0, logit for C is 0
    model.lm_head.weight.copy_(torch.eye(vocab_size))
    model.lm_head.bias.zero_()


# ----------------------------------------------------------------------------
# test on all 27 possible 3-character inputs (A/B/C at each position)
# ----------------------------------------------------------------------------

model.eval()

print("Testing copy task on all 27 inputs (3 tokens, 3 positions):\n")

correct = 0
total   = 0

for a in range(3):
    for b in range(3):
        for c in range(3):
            inp = torch.tensor([[a, b, c]], device=device)  # shape (1, 3)
            logits = model(inp)                              # shape (1, 3, vocab_size)

            # predicted token at each position = argmax of logits
            preds = logits.argmax(dim=-1)[0].tolist()       # list of 3 predicted indices

            input_str  = ''.join(itos[i] for i in [a, b, c])
            output_str = ''.join(itos[i] for i in preds)
            ok = '✓' if preds == [a, b, c] else '✗'

            print(f"  input: {input_str}  output: {output_str}  {ok}")

            if preds == [a, b, c]:
                correct += 1
            total += 1

print(f"\nResult: {correct}/{total} correct")
