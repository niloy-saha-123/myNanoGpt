import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# Hand-designed Autoregressive GPT: 1-Digit Decimal Addition
#
# Goal: given "a + b =", generate the 2-digit base-10 result "d1 d0"
# where d1 = (a+b) // 10  and  d0 = (a+b) % 10, one token at a time.
#
# Examples:
#   7 + 8 = 1 5   (15 = "15")
#   3 + 4 = 0 7   (7 = "07")
#   9 + 9 = 1 8   (18 = "18")
#   0 + 0 = 0 0
#
# Sequence layout:
#   pos:  0   1   2   3   4   5
#   tok:  a   +   b   =  d1  d0
#
# Step 1: model sees [a, +, b, =]      and predicts d1 (tens digit, 0 or 1)
# Step 2: model sees [a, +, b, =, d1]  and predicts d0 (ones digit, 0-9)
#
# Attention routes operand token indicators to
# the output position, and lm_head weights are calibrated from the known
# post-LayerNorm vectors for all 200 cases (100 inputs x 2 steps).
# ----------------------------------------------------------------------------

BASE = 10
vocab_size = 12  # digits 0-9, +, =
n_embd = 25      # 12 token dims + 2 operand markers + 1 output marker + 10 value dims
n_head = 1
n_layer = 1
block_size = 6   # a + b = d1 d0
dropout = 0.0
device = 'cpu'

itos = {i: str(i) for i in range(10)}
itos[10] = '+'
itos[11] = '='
stoi = {v: k for k, v in itos.items()}

# ---------------------------------------------------------------------------
# Model definition (same architecture as all other hand-designed GPTs)
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
# Set weights by hand
#
# Embedding layout (25 dims):
#   dims 0-11:  token one-hot (0-9, +, =)
#   dim 12:     position 0 marker (operand a)
#   dim 13:     position 2 marker (operand b)
#   dim 14:     output position marker (pos 3 AND pos 4)
#   dims 15-24: value dims for digit token indicators (filled by attention V)
#
# Attention routes operand info to output positions (same as ternary).
# FFN is zeroed. lm_head is calibrated from post-LN_f vectors.
# ---------------------------------------------------------------------------
model = GPT().to(device)

with torch.no_grad():
    # token embedding: one-hot in first vocab_size dims
    tok_emb = torch.zeros(vocab_size, n_embd)
    for i in range(vocab_size):
        tok_emb[i, i] = 1.0
    model.token_embedding_table.weight.copy_(tok_emb)

    # position embedding
    pos_emb = torch.zeros(block_size, n_embd)
    pos_emb[0, 12] = 1.0   # operand a
    pos_emb[2, 13] = 1.0   # operand b
    pos_emb[3, 14] = 1.0   # output position
    pos_emb[4, 14] = 1.0   # output position (same marker)
    model.position_embedding_table.weight.copy_(pos_emb)

    head = model.blocks[0].sa.heads[0]

    # Q: output marker (dim 14) -> query dims 12, 13
    W_Q = torch.zeros(n_embd, n_embd)
    W_Q[12, 14] = 10.0
    W_Q[13, 14] = 10.0
    head.query.weight.copy_(W_Q)

    # K: operand markers pass through
    W_K = torch.zeros(n_embd, n_embd)
    W_K[12, 12] = 10.0
    W_K[13, 13] = 10.0
    head.key.weight.copy_(W_K)

    # V: extract digit indicators into value dims
    #    token d (dim d, for d=0..9) -> dim 15+d
    W_V = torch.zeros(n_embd, n_embd)
    for d in range(BASE):
        W_V[15 + d, d] = 1.0
    head.value.weight.copy_(W_V)

    # projection: pass value dims into residual
    W_proj = torch.zeros(n_embd, n_embd)
    for d in range(BASE):
        W_proj[15 + d, 15 + d] = 1.0
    model.blocks[0].sa.proj.weight.copy_(W_proj)
    model.blocks[0].sa.proj.bias.zero_()

    # FFN: zeroed out
    model.blocks[0].ffwd.net[0].weight.zero_()
    model.blocks[0].ffwd.net[0].bias.zero_()
    model.blocks[0].ffwd.net[2].weight.zero_()
    model.blocks[0].ffwd.net[2].bias.zero_()

    # LayerNorms: identity affine
    model.blocks[0].ln1.weight.fill_(1.0); model.blocks[0].ln1.bias.zero_()
    model.blocks[0].ln2.weight.fill_(1.0); model.blocks[0].ln2.bias.zero_()
    model.ln_f.weight.fill_(1.0); model.ln_f.bias.zero_()

    # -----------------------------------------------------------------------
    # Calibrate lm_head from post-LN_f vectors for all 200 cases
    # -----------------------------------------------------------------------
    model.eval()
    vectors = []
    targets = []

    for a in range(BASE):
        for b in range(BASE):
            s = a + b
            d1 = s // BASE
            d0 = s % BASE

            # step 1: predict d1
            ctx1 = [a, stoi['+'], b, stoi['=']]
            idx1 = torch.tensor([ctx1], device=device)
            tok = model.token_embedding_table(idx1)
            pos = model.position_embedding_table(torch.arange(len(ctx1), device=device))
            x = tok + pos
            x = model.blocks(x)
            x = model.ln_f(x)
            vectors.append(x[0, -1].clone())
            targets.append(d1)

            # step 2: predict d0
            ctx2 = [a, stoi['+'], b, stoi['='], d1]
            idx2 = torch.tensor([ctx2], device=device)
            tok = model.token_embedding_table(idx2)
            pos = model.position_embedding_table(torch.arange(len(ctx2), device=device))
            x = tok + pos
            x = model.blocks(x)
            x = model.ln_f(x)
            vectors.append(x[0, -1].clone())
            targets.append(d0)

    X = torch.stack(vectors)  # (200, n_embd)

    # find lm_head weights that correctly classify all 200 cases
    target_tensor = torch.tensor(targets, device=device)
    W = torch.zeros(vocab_size, n_embd)
    b = torch.zeros(vocab_size)
    lr = 1.0
    for step in range(20000):
        logits = X @ W.T + b
        if (logits.argmax(dim=1) == target_tensor).all():
            break
        loss = F.cross_entropy(logits, target_tensor)
        probs = F.softmax(logits, dim=1)
        grad = probs.clone()
        for i in range(len(targets)):
            grad[i, targets[i]] -= 1.0
        grad /= len(targets)
        W -= lr * (grad.T @ X)
        b -= lr * grad.sum(dim=0)

    model.lm_head.weight.copy_(W)
    model.lm_head.bias.copy_(b)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_sequence(context):
    model.eval()
    idx = torch.tensor([context], device=device)
    generated = []
    for _ in range(2):
        logits = model(idx)
        next_token = logits[0, -1, :].argmax().item()
        idx = torch.cat((idx, torch.tensor([[next_token]], device=device)), dim=1)
        generated.append(next_token)
    return generated

# ---------------------------------------------------------------------------
# Test all 100 inputs
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing autoregressive 1-digit decimal addition:\n")

    correct = 0
    total = 0
    for a in range(BASE):
        for b in range(BASE):
            s = a + b
            d1 = s // BASE
            d0 = s % BASE
            context = [a, stoi['+'], b, stoi['=']]
            preds = generate_sequence(context)

            for step in range(1, len(preds) + 1):
                so_far = ''.join(itos[t] for t in preds[:step])
                print(f"Input: {a}+{b}=  Output: {so_far}")

            if preds == [d1, d0]:
                correct += 1
            total += 1
            print()

    print(f"{correct}/{total} correct")
