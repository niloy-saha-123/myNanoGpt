import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# Hand-designed Autoregressive GPT: 1-Digit Binary Addition
#
# Goal: given "a + b =", predict the sum (0, 1, or 2) autoregressively.
#
# Sequence layout:
#   pos:  0   1   2   3   4
#   tok:  a   +   b   =  sum
#
# The '=' token at pos 3 must attend to pos 0 and pos 2 (the operands),
# figure out how many are '1', and predict the correct sum token.
#
# Challenge: LayerNorm normalizes to zero-mean/unit-variance, so storing
# the sum as a single magnitude (e.g. 0/50/100) doesn't work — LN maps
# both 50 and 100 to ~2.65.  Instead, we extract TWO value dims (one per
# token indicator) so the three sum values create distinct *patterns*
# across dimensions that survive normalization.
#
# The FFN then classifies these patterns using (dim1 − dim0) as a
# discriminant with ReLU thresholding.
# ----------------------------------------------------------------------------

vocab_size = 5   # 0,1,2,+,=
# Embedding dimension: 5 token one‑hots + 3 extra slots for position markers/value
n_embd = 8
n_head = 1
n_layer = 1
block_size = 5
dropout = 0.0
device = 'cpu'

itos = {0: '0', 1: '1', 2: '2', 3: '+', 4: '='}
stoi = {'0': 0, '1': 1, '2': 2, '+': 3, '=': 4}

# ---------------------------------------------------------------------------
# Model definition (identical to the tiny‑GPT used elsewhere)
# ---------------------------------------------------------------------------
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
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(num_heads * head_size, n_embd)
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
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa   = MultiHeadAttention(n_head, n_embd // n_head)
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
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

# ---------------------------------------------------------------------------
# Manual weight setting
#
# Key insight: LayerNorm normalizes to zero-mean / unit-variance, so a single
# value dimension can't distinguish sum=1 from sum=2 (both get normalized to
# ~2.65).  Instead we extract TWO value dims — one for token '0' and one for
# token '1'.  After attention averages over the two operand positions, the
# three cases produce distinct *patterns* across the two dims:
#
#   0+0 → (high, low)    i.e. dim0 >> dim1
#   0+1 → (mid,  mid)    i.e. dim0 ≈ dim1
#   1+1 → (low,  high)   i.e. dim1 >> dim0
#
# These patterns survive LayerNorm because LN preserves relative ordering.
# The FFN then classifies using (dim1 − dim0) as the discriminant.
# ---------------------------------------------------------------------------
model = GPT().to(device)
with torch.no_grad():
    # Token embedding: one-hot identity in the first `vocab_size` columns
    embed = torch.zeros(vocab_size, n_embd)
    embed[torch.arange(vocab_size), torch.arange(vocab_size)] = 1.0
    model.token_embedding_table.weight.copy_(embed)

    # Position embedding: mark operand positions so attention can find them
    pos_emb = torch.zeros(block_size, n_embd)
    pos_emb[0, 5] = 1.0   # marker for position 0 (first operand)
    pos_emb[2, 6] = 1.0   # marker for position 2 (second operand)
    model.position_embedding_table.weight.copy_(pos_emb)

    head = model.blocks[0].sa.heads[0]

    # Q: the '=' token (dim 4) produces query in dims 5,6 to match K's markers
    # weight[out, in]: input dim 4 → output dims 5, 6
    W_Q = torch.zeros(n_embd, n_embd)
    W_Q[5, 4] = 10.0
    W_Q[6, 4] = 10.0
    head.query.weight.copy_(W_Q)

    # K: position markers pass through (dim 5→5, dim 6→6)
    W_K = torch.zeros(n_embd, n_embd)
    W_K[5, 5] = 10.0
    W_K[6, 6] = 10.0
    head.key.weight.copy_(W_K)

    # V: extract BOTH token indicators into separate value dims
    # After ln1, dim 0 is positive for token '0', dim 1 is positive for token '1'
    # weight[out, in]: input dim 0 → v[0],  input dim 1 → v[1]
    W_V = torch.zeros(n_embd, n_embd)
    W_V[0, 0] = 1.0
    W_V[1, 1] = 1.0
    head.value.weight.copy_(W_V)

    # Projection: pass value dims 0,1 into residual dims 0,1
    # (the '=' token has 0 in dims 0,1, so these don't collide)
    W_proj = torch.zeros(n_embd, n_embd)
    W_proj[0, 0] = 1.0
    W_proj[1, 1] = 1.0
    model.blocks[0].sa.proj.weight.copy_(W_proj)
    model.blocks[0].sa.proj.bias.zero_()

    # -------------------------------------------------------------------
    # FFN: classify based on (dim1 − dim0) after ln2
    #
    # After ln2, the three cases give roughly:
    #   0+0: dim0 ≈ +2.14, dim1 ≈ −1.24  →  diff ≈ −3.37
    #   0+1: dim0 ≈ +0.83, dim1 ≈ +0.83  →  diff ≈  0.00
    #   1+1: dim0 ≈ −1.24, dim1 ≈ +2.14  →  diff ≈ +3.37
    #
    # Three ReLU neurons threshold on this difference:
    #   n0 = ReLU(dim1 − dim0 + 1.5)   fires for sum ≥ 1
    #   n1 = ReLU(dim1 − dim0 − 1.5)   fires for sum = 2
    #   n2 = ReLU(dim0 − dim1 − 1.5)   fires for sum = 0
    # -------------------------------------------------------------------
    hidden = 4 * n_embd
    W1 = torch.zeros(n_embd, hidden)
    b1 = torch.zeros(hidden)

    # n0: fires when sum >= 1  (dim1 − dim0 > −1.5)
    W1[1, 0] =  1.0;  W1[0, 0] = -1.0;  b1[0] =  1.5
    # n1: fires when sum = 2   (dim1 − dim0 > +1.5)
    W1[1, 1] =  1.0;  W1[0, 1] = -1.0;  b1[1] = -1.5
    # n2: fires when sum = 0   (dim0 − dim1 > +1.5)
    W1[0, 2] =  1.0;  W1[1, 2] = -1.0;  b1[2] = -1.5

    model.blocks[0].ffwd.net[0].weight.copy_(W1.T)
    model.blocks[0].ffwd.net[0].bias.copy_(b1)

    # Second layer: map neuron activations → token logits (dims 0,1,2)
    W2 = torch.zeros(hidden, n_embd)
    W2[0, 0] = -100.0;  W2[0, 1] =  100.0                   # n0: suppress '0', promote '1'
    W2[1, 1] = -500.0;  W2[1, 2] =  500.0                   # n1: suppress '1', promote '2'
    W2[2, 0] =  100.0                                        # n2: promote '0'
    model.blocks[0].ffwd.net[2].weight.copy_(W2.T)
    model.blocks[0].ffwd.net[2].bias.zero_()

    # LayerNorms: weight=1, bias=0 (affine part is identity)
    model.blocks[0].ln1.weight.fill_(1.0); model.blocks[0].ln1.bias.zero_()
    model.blocks[0].ln2.weight.fill_(1.0); model.blocks[0].ln2.bias.zero_()
    model.ln_f.weight.fill_(1.0); model.ln_f.bias.zero_()

    # Unembedding: map dims 0,1,2 to logits for tokens '0','1','2'
    W_out = torch.zeros(vocab_size, n_embd)
    W_out[0, 0] = 1.0
    W_out[1, 1] = 1.0
    W_out[2, 2] = 1.0
    model.lm_head.weight.copy_(W_out)
    model.lm_head.bias.zero_()

# ---------------------------------------------------------------------------
# Generation helper (autoregressive)
# ---------------------------------------------------------------------------
def generate_sequence(context):
    model.eval()
    idx = torch.tensor([context], device=device)
    logits = model(idx)
    next_token = logits[0, -1, :].argmax().item()
    return next_token

# test on all 4 inputs
if __name__ == '__main__':
    print("Testing autoregressive 1-digit binary addition:\n")

    correct = 0
    total = 0
    for a in [0, 1]:
        for b in [0, 1]:
            context = [stoi[str(a)], stoi['+'], stoi[str(b)], stoi['=']]
            expected = a + b
            pred = generate_sequence(context)
            print(f"Input: {a}+{b}=  Output: {itos[pred]}")
            if pred == stoi[str(expected)]:
                correct += 1
            total += 1

    print(f"\n{correct}/{total} correct")
