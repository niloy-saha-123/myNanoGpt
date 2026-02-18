import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# Hand-designed GPT: Reverse Task
#
# Goal: given input A B C, output C B A (reverse the sequence)
#
# This one actually needs real attention. The output at position 0 has to
# read from input position 2, position 1 reads from position 1, and position
# 2 reads from position 0. That's an anti-diagonal attention pattern.
#
# How we build the attention pattern by hand:
#   - n_embd = 6: first 3 dims = token content, last 3 dims = position info
#   - Token embedding: A->[1,0,0,0,0,0], B->[0,1,0,0,0,0], C->[0,0,1,0,0,0]
#   - Position embedding adds to last 3 dims:
#       pos 0 -> [0,0,0, 1,0,0]
#       pos 1 -> [0,0,0, 0,1,0]
#       pos 2 -> [0,0,0, 0,0,1]
#   - So the full input vector at position i is: [token_onehot | pos_onehot]
#
#   - W_Q (3x6 -> 3): extracts the position part [dims 3,4,5]
#     Q[i] = pos_onehot of position i
#
#   - W_K (3x6 -> 3): extracts the FLIPPED position part
#     K[j] = flipped_pos_onehot of position j
#     flip means: pos 0 -> [0,0,1], pos 1 -> [0,1,0], pos 2 -> [1,0,0]
#
#   - Q[i] · K[j] is large (=1) exactly when i + j = 2, i.e. j = 2-i
#     This gives us the anti-diagonal attention pattern we want!
#
#   - W_V: extracts the token content part [dims 0,1,2]
#     V[j] = token_onehot of position j
#
#   - After attention, each position i holds the token content of position 2-i
#   - Unembedding maps that token content back to the correct output token
# ----------------------------------------------------------------------------

vocab_size = 3   # A=0, B=1, C=2
n_embd     = 6   # 3 for token content + 3 for position info
n_head     = 1
n_layer    = 1
block_size = 3
dropout    = 0.0

device = 'cpu'

itos = {0: 'A', 1: 'B', 2: 'C'}
stoi = {'A': 0, 'B': 1, 'C': 2}

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
        # scale by head_size not C, since head_size != n_embd here
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size**-0.5
        # NOTE: we remove the causal mask here -- reverse needs to look forward too
        # so we allow full attention (no masking)
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
    # token embedding: A/B/C -> one-hot in first 3 dims, last 3 dims are zero
    # shape: (vocab_size=3, n_embd=6)
    tok_emb = torch.zeros(vocab_size, n_embd)
    tok_emb[0, 0] = 1.0  # A -> [1,0,0, 0,0,0]
    tok_emb[1, 1] = 1.0  # B -> [0,1,0, 0,0,0]
    tok_emb[2, 2] = 1.0  # C -> [0,0,1, 0,0,0]
    model.token_embedding_table.weight.copy_(tok_emb)

    # position embedding: pos i -> one-hot in last 3 dims, first 3 dims are zero
    # shape: (block_size=3, n_embd=6)
    pos_emb = torch.zeros(block_size, n_embd)
    pos_emb[0, 3] = 1.0  # pos 0 -> [0,0,0, 1,0,0]
    pos_emb[1, 4] = 1.0  # pos 1 -> [0,0,0, 0,1,0]
    pos_emb[2, 5] = 1.0  # pos 2 -> [0,0,0, 0,0,1]
    model.position_embedding_table.weight.copy_(pos_emb)

    # head size = n_embd // n_head = 6 // 1 = 6
    head = model.blocks[0].sa.heads[0]

    # W_Q: extracts position dims [3,4,5] from the input vector
    # shape: (n_embd=6, head_size=6)
    W_Q = torch.zeros(n_embd, n_embd)
    W_Q[3, 3] = 1.0
    W_Q[4, 4] = 1.0
    W_Q[5, 5] = 1.0
    head.query.weight.copy_(W_Q)

    # W_K: extracts FLIPPED position dims from the input vector
    # pos 0 ([...,1,0,0]) should produce key [0,0,1]
    # pos 1 ([...,0,1,0]) should produce key [0,1,0]
    # pos 2 ([...,0,0,1]) should produce key [1,0,0]
    # so: key[3] = input[5], key[4] = input[4], key[5] = input[3]
    W_K = torch.zeros(n_embd, n_embd)
    W_K[3, 5] = 1.0
    W_K[4, 4] = 1.0
    W_K[5, 3] = 1.0
    head.key.weight.copy_(W_K)

    # W_V: extracts token content dims [0,1,2] from the input vector
    # so the value at position j is just the token one-hot of position j
    W_V = torch.zeros(n_embd, n_embd)
    W_V[0, 0] = 1.0
    W_V[1, 1] = 1.0
    W_V[2, 2] = 1.0
    head.value.weight.copy_(W_V)

    # attention output projection: pass through the first 3 dims (token content),
    # zero out the position dims since we don't need them after attention
    W_proj = torch.zeros(n_embd, n_embd)
    W_proj[0, 0] = 1.0
    W_proj[1, 1] = 1.0
    W_proj[2, 2] = 1.0
    model.blocks[0].sa.proj.weight.copy_(W_proj)
    model.blocks[0].sa.proj.bias.zero_()

    # FFN: zero out (no computation needed after attention)
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

    # unembedding: maps token content dims [0,1,2] back to vocab logits
    # if dim 0 is highest -> predict A, dim 1 -> B, dim 2 -> C
    # shape: (vocab_size=3, n_embd=6)
    W_out = torch.zeros(vocab_size, n_embd)
    W_out[0, 0] = 1.0  # logit for A comes from dim 0
    W_out[1, 1] = 1.0  # logit for B comes from dim 1
    W_out[2, 2] = 1.0  # logit for C comes from dim 2
    model.lm_head.weight.copy_(W_out)
    model.lm_head.bias.zero_()


# ----------------------------------------------------------------------------
# test on all 27 possible 3-character inputs
# ----------------------------------------------------------------------------

model.eval()

print("Testing reverse task on all 27 inputs (3 tokens, 3 positions):\n")

correct = 0
total   = 0

for a in range(3):
    for b in range(3):
        for c in range(3):
            inp = torch.tensor([[a, b, c]], device=device)
            logits = model(inp)                        # (1, 3, vocab_size)
            preds  = logits.argmax(dim=-1)[0].tolist() # predicted token at each position

            expected = [c, b, a]  # reversed!
            input_str    = ''.join(itos[i] for i in [a, b, c])
            output_str   = ''.join(itos[i] for i in preds)
            expected_str = ''.join(itos[i] for i in expected)
            ok = '✓' if preds == expected else '✗'

            print(f"  input: {input_str}  expected: {expected_str}  output: {output_str}  {ok}")

            if preds == expected:
                correct += 1
            total += 1

print(f"\nResult: {correct}/{total} correct")
