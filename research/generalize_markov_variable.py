"""
Generalization / Fidelity Test: Markov Chain (variable training size)

Same as generalize_markov but with configurable num_train_seqs.
Compare: 256 vs 512 vs 1024 vs 4096 training sequences.
Lower KL/TV = better match to true transition matrix.

Usage:
  python generalize_markov_variable.py           # default 256
  python generalize_markov_variable.py 1024     # 1024 sequences
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import random
from model import GPT

# ---------- config ----------
vocab_size = 3
n_embd     = 6
n_head     = 1
n_layer    = 1
block_size = 16
device     = 'cpu' if os.environ.get('RESEARCH_USE_CPU') else ('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

itos = {0: 'A', 1: 'B', 2: 'C'}
stoi = {v: k for k, v in itos.items()}

P = torch.tensor([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.4, 0.4, 0.2],
], dtype=torch.float)

lr         = 1e-3
max_iters  = 6000   # more iters when using more data
eval_every = 500
train_seq_len = 16

# Parse num_train_seqs from command line
num_train_seqs = int(sys.argv[1]) if len(sys.argv) > 1 else 256


def sample_chain(start, length):
    x = [start]
    for _ in range(length - 1):
        probs = P[x[-1]]
        nxt = torch.multinomial(probs, num_samples=1).item()
        x.append(nxt)
    return x


def make_dataset(num_seqs, length):
    xs, ys = [], []
    for _ in range(num_seqs):
        start = random.choice([0, 1, 2])
        seq = sample_chain(start, length)
        xs.append(seq[:-1])
        ys.append(seq[1:])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def empirical_transition(model, num_seqs=200, length=16):
    model.eval()
    counts = torch.zeros_like(P)
    with torch.no_grad():
        for _ in range(num_seqs):
            s = random.choice([0, 1, 2])
            idx = torch.tensor([[s]], device=device)
            last = s
            for _ in range(length - 1):
                logits, _ = model(idx)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                nxt = torch.multinomial(probs, num_samples=1).item()
                counts[last, nxt] += 1
                idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
                last = nxt
    row_sums = counts.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return counts / row_sums


def kl_and_tv(p_true, p_est):
    kl = (p_true * (p_true.clamp_min(1e-9).log() - p_est.clamp_min(1e-9).log())).sum(dim=1)
    tv = 0.5 * (p_true - p_est).abs().sum(dim=1)
    return kl, tv


def build_hand_markov():
    m = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    with torch.no_grad():
        tok_emb = torch.zeros(vocab_size, n_embd)
        for i in range(vocab_size):
            tok_emb[i, i] = 1.0
        m.token_embedding_table.weight.copy_(tok_emb)
        m.position_embedding_table.weight.zero_()
        for block in m.blocks:
            for head in block.sa.heads:
                head.key.weight.zero_()
                head.query.weight.zero_()
                head.value.weight.zero_()
            block.sa.proj.weight.zero_()
            block.sa.proj.bias.zero_()
            block.ffwd.net[0].weight.zero_()
            block.ffwd.net[0].bias.zero_()
            block.ffwd.net[2].weight.zero_()
            block.ffwd.net[2].bias.zero_()
            block.ln1.weight.fill_(1.0)
            block.ln1.bias.zero_()
            block.ln2.weight.fill_(1.0)
            block.ln2.bias.zero_()
        m.ln_f.weight.fill_(1.0)
        m.ln_f.bias.zero_()
        m.lm_head.weight.zero_()
        m.lm_head.weight[:, :3] = P.log().T
        if hasattr(m.lm_head, 'bias') and m.lm_head.bias is not None:
            m.lm_head.bias.zero_()
    return m


def evaluate_chain(model, label, num_seqs=200, length=16):
    est = empirical_transition(model, num_seqs=num_seqs, length=length)
    kl, tv = kl_and_tv(P, est)
    print(f"\n{label}")
    print("  Estimated transition matrix:")
    print(est.cpu().numpy())
    print("  KL per row: ", kl.cpu().numpy())
    print("  TV per row: ", tv.cpu().numpy())
    print(f"  KL mean={kl.mean().item():.4f}  TV mean={tv.mean().item():.4f}")
    return est, kl, tv


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 60)
    print(f"  MARKOV CHAIN FIDELITY (train_seqs={num_train_seqs})")
    print("=" * 60)

    hand_model = build_hand_markov()
    train_x, train_y = make_dataset(num_train_seqs, train_seq_len)
    train_x, train_y = train_x.to(device), train_y.to(device)
    trained_model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    opt = torch.optim.AdamW(trained_model.parameters(), lr=lr)

    print(f"Training GPT on {num_train_seqs} sequences of length {train_seq_len}...\n")
    for step in range(max_iters + 1):
        logits, loss = trained_model(train_x, train_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("  TRANSITION MATRIX COMPARISON")
    print("=" * 60)

    evaluate_chain(hand_model, "Hand-Designed Model")
    evaluate_chain(trained_model, "Trained (SGD) Model")
