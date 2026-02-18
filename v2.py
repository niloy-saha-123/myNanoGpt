import torch
import torch.nn as nn
from torch.nn import functional as F


#hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500 # how often should we evaluate the loss?
learning_rate = 3e-3 # how much to update the parameters?
n_embd = 384 # how many embedding dimensions to use?
n_head = 6 # how many heads to use in the self-attention layers?
n_layer = 6 # how many layers to use in the transformer?
dropout = 0.2 # dropout rate

# prefer Apple Silicon GPU (Metal) on macOS, then CUDA, then CPU
device = (
    'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)
eval_iters = 200 # how many iterations to evaluate the loss?

# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars) # how many unique characters are there?
#create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # this is the encoder, take a string, output a list of integers. 
decode = lambda l: ''.join([itos[i] for i in l]) # this is the decoder, take a list of integers, output a string.

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data # this is the data to get the batch from
    ix = torch.randint(len(data) - block_size, (batch_size,)) # this is the random index to get the data
    x = torch.stack([data[i:i+block_size] for i in ix]) # this is the input data
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # this is the target data
    x, y = x.to(device), y.to(device) # move the data to the device
    return x, y # return the input and target data

# evaluate the loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']: # this is the split to evaluate the loss on
        losses = torch.zeros(eval_iters) # this is the loss for the split
        for k in range(eval_iters): # this is the number of iterations to evaluate the loss
            X, Y = get_batch(split) # this is the batch to evaluate the loss on
            logits, loss = model(X, Y) # this is the loss for the batch
            losses[k] = loss.item() # this is the loss for the batch
        out[split] = losses.mean() # this is the average loss for the split
    model.train() # this is the model to train
    return out # return the average loss for the split

# single head of self-attention
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -----> (B, T, T) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -----> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# feed forward network
# it's a simple linear layer followed by a non-linearity
# what it does is it takes the output of the self-attention heads and projects it to the vocabulary size
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # dropout to prevent overfitting
        )
        
    def forward(self, x):
        return self.net(x)

# transformer block is a block of self-attention and a feed forward network
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads
        super().__init__()
        head_size = n_embd // n_head # number of heads
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # (B,T,n_embd)
        x = x + self.ffwd(self.ln2(x)) # (B,T,n_embd)
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # n_embd is the number of embedding dimensions
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_heads = Head(n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) ## i.e. 4 heads of 8 dimensions each (8 self attention heads)
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # this is the linear layer to project the embedding dimensions to the vocabulary size

    # get the logits for the next token
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) where each index in batch is a (B,T) tensor of indices
        # logits shape is (B,T,C) where C is the number of channels
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,n_embd) apply the transformer blocks
        logits = self.lm_head(x) # (B,T,vocab_size) vocab_size = C but not equal to the C above because the one above is embed

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step (B, T, C) -> (B, C)
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append the sampled token index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# create a PyTorch model
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# train the model
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0: # this is the interval to evaluate the loss on
        losses = estimate_loss() # this is the loss for the train and val sets
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") # this is the loss for the train and val sets

    # sample a batch of data
    xb, yb = get_batch('train') # this is the batch to train on
    # evaluate the loss
    logits, loss = model(xb, yb) # this is the loss for the batch
    optimizer.zero_grad(set_to_none=True) # this is the optimizer to zero the gradients
    loss.backward() # this is the loss to backward
    optimizer.step() # this is the optimizer to step

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # this is the initial context
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # this is the generated text