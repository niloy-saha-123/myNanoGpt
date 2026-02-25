import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# Hand-designed Autoregressive GPT: Markov Chain
#
# Goal: simulate a Markov Chain over 3 states (A, B, C).
# The sequence is generated token by token, where the probability of the NEXT
# token depends ONLY on the CURRENT token.
#
# This means:
# - Attention is irrelevant (or it just looks at the current token position).
# - FFN is irrelevant.
# - The Unembedding matrix (lm_head) stores the transition probabilities!
#
# Let's define the transition matrix T where T[i,j] is the log-probability
# of transitioning from state i to state j:
#
#       Next: A    B    C
# Cur: A     0.7  0.2  0.1  --> log probs
#      B     0.1  0.8  0.1
#      C     0.4  0.4  0.2
# ----------------------------------------------------------------------------

vocab_size = 3   # A=0, B=1, C=2
n_embd     = 3   # 3 for token content (no position info needed for Markov!)
n_head     = 1
n_layer    = 1
block_size = 32
dropout    = 0.0
device = 'cpu'

itos = {0: 'A', 1: 'B', 2: 'C'}
stoi = {'A': 0, 'B': 1, 'C': 2}

# Transition Matrix (Probabilities)
P = torch.tensor([
    [0.7, 0.2, 0.1],  # From A
    [0.1, 0.8, 0.1],  # From B
    [0.4, 0.4, 0.2]   # From C
])
log_P = torch.log(P)

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
        # causal mask
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
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        # Position embedding is zeros because Markov only cares about current state!
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
    # token embedding: identity
    model.token_embedding_table.weight.copy_(torch.eye(vocab_size))

    # position embedding: ALL ZEROS. Markov doesn't care about position!
    model.position_embedding_table.weight.zero_()

    for block in model.blocks:
        for head in block.sa.heads:
            # We want position t to ONLY attend to position t.
            # Using identity matrix on Q,K means dot product is 1.0 for same token, 0 for others.
            # Because of the casual mask & softmax, if we pass nothing to Q/K (all zeros), 
            # attention becomes an average of the past. That's WRONG for Markov!
            # We must design Q and K so that current pos attends heavily to ITSELF.
            # Actually, the simplest approach: make Q and K identical to positional embeddings 
            # and scale them to infinity.
            # Or even better... disable attention completely by making V zero, and passing 
            # the token embedding straight through the residual stream!
            head.key.weight.zero_()
            head.query.weight.zero_()
            head.value.weight.zero_()
        
        block.sa.proj.weight.zero_()
        block.sa.proj.bias.zero_()
        block.ffwd.net[0].weight.zero_()
        block.ffwd.net[0].bias.zero_()
        block.ffwd.net[2].weight.zero_()
        block.ffwd.net[2].bias.zero_()

        # LayerNorms: identity
        block.ln1.weight.fill_(1.0); block.ln1.bias.zero_()
        block.ln2.weight.fill_(1.0); block.ln2.bias.zero_()
        
    model.ln_f.weight.fill_(1.0); model.ln_f.bias.zero_()

    # unembedding: The magic happens here!
    # If the residual stream has token A [1,0,0], W_out must produce log_P[0,:]
    # W_out matrix must be exactly log_P transposed!
    model.lm_head.weight.copy_(log_P.T)
    model.lm_head.bias.zero_()

# ----------------------------------------------------------------------------
# Sequence generation function
# ----------------------------------------------------------------------------

def generate_sequence(start_token, length=10):
    model.eval()
    idx = torch.tensor([[start_token]], device=device)
    for _ in range(length):
        logits = model(idx)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        idx = torch.cat((idx, torch.tensor([[next_token]], device=device)), dim=1)
    return idx[0].tolist()

# test the markov chain
if __name__ == '__main__':
    print("Testing autoregressive Markov chain generation:")
    print("Transition matrix:")
    print(P.numpy())
    print()

    # generate a sequence and show it growing token by token
    print("Generating 20 tokens starting from A:\n")
    start = stoi['A']
    seq = generate_sequence(start, length=20)
    for i in range(1, len(seq) + 1):
        print(''.join(itos[t] for t in seq[:i]))
