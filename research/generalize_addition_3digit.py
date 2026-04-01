"""
Generalization Test: 3-Digit Decimal Addition

Two 3-digit numbers: a b c + d e f = (4 digits for sum, max 999+999=1998)
e.g. 456 + 789 = 1245

Input space: 1000 * 1000 = 1,000,000 possible pairs.
We can train on 1k, 5k, 10k, etc. and test on held-out.

Usage:
  python generalize_addition_3digit.py           # default 5000 train / 2000 test
  python generalize_addition_3digit.py 1000       # 1k train
  python generalize_addition_3digit.py 10000      # 10k train
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import os
import random
from model import GPT

# ---------- config ----------
vocab_size = 12   # 0-9, +, =
n_embd     = 64   # larger for 3-digit
n_head     = 2
n_layer    = 1
# seq: abc + def = ghij  -> 3+1+3+1+4 = 12 tokens
block_size = 12

# ---------- configurable train/test sizes (via cmd line) ----------
TRAIN_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
# Use CPU for large batches to avoid MPS OOM (>2k examples on Apple Silicon)
_use_cpu = os.environ.get('RESEARCH_USE_CPU') or TRAIN_SIZE > 2000
device   = 'cpu' if _use_cpu else ('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

itos = {i: str(i) for i in range(10)}
itos[10] = '+'
itos[11] = '='
stoi = {v: k for k, v in itos.items()}

lr         = 1e-3
max_iters  = 30000
eval_every = 3000
TEST_SIZE  = 2000
TOTAL_PAIRS = 1000 * 1000

# ---------- build random train/test splits ----------
random.seed(42)
all_pairs = [(a, b) for a in range(1000) for b in range(1000)]
random.shuffle(all_pairs)
train_examples = all_pairs[:TRAIN_SIZE]
test_examples  = all_pairs[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]


def add_3digit(a, b):
    """a, b in 0..999. Return 4 digits: thousands, hundreds, tens, ones."""
    s = a + b  # max 1998
    d3 = (s // 1000) % 10
    d2 = (s // 100) % 10
    d1 = (s // 10) % 10
    d0 = s % 10
    return [d3, d2, d1, d0]


def to_digits(n, pad=3):
    """Convert int to list of digits, padded to pad length."""
    if n == 0:
        return [0] * pad
    digs = []
    while n:
        digs.append(n % 10)
        n //= 10
    digs.reverse()
    return [0] * (pad - len(digs)) + digs


def make_dataset(examples):
    inputs, targets = [], []
    for (a, b) in examples:
        a_d = to_digits(a, 3)  # [a2, a1, a0]
        b_d = to_digits(b, 3)  # [b2, b1, b0]
        out = add_3digit(a, b)  # [d3, d2, d1, d0]
        # seq: a2 a1 a0 + b2 b1 b0 = d3 d2 d1 d0
        seq = a_d + [stoi['+']] + b_d + [stoi['=']] + out
        inp = seq[:-1]   # up to before last output digit
        tgt = seq[1:]
        # Mask: only care about output digits (d3,d2,d1,d0 at tgt indices 7,8,9,10)
        masked_tgt = [-100] * len(tgt)
        for i in [7, 8, 9, 10]:
            masked_tgt[i] = tgt[i]
        inputs.append(inp)
        targets.append(masked_tgt)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def evaluate(model, examples):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (a, b) in examples:
            a_d = to_digits(a, 3)
            b_d = to_digits(b, 3)
            expected = add_3digit(a, b)
            context = a_d + [stoi['+']] + b_d + [stoi['=']]
            idx = torch.tensor([context], device=device)
            preds = []
            for _ in range(4):
                logits, _ = model(idx)
                next_tok = logits[0, -1, :].argmax().item()
                idx = torch.cat([idx, torch.tensor([[next_tok]], device=device)], dim=1)
                preds.append(next_tok)
            ok = preds == expected
            if ok:
                correct += 1
            total += 1
    model.train()
    return correct, total


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 60)
    print("  GENERALIZATION TEST: 3-Digit Decimal Addition")
    print("=" * 60)
    print(f"\n  Total possible: {TOTAL_PAIRS:,}")
    print(f"  Train set: {len(train_examples):,} examples")
    print(f"  Test set:  {len(test_examples):,} examples (NEVER seen)")
    print("  (No hand-designed model — focus on trained generalization)")

    trained_model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    optimizer = torch.optim.AdamW(trained_model.parameters(), lr=lr)
    train_inputs, train_targets = make_dataset(train_examples)
    train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)

    print(f"\nTraining on {len(train_examples):,} examples...\n")
    for step in range(max_iters + 1):
        logits, loss = trained_model(train_inputs, train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            tr_c, tr_t = evaluate(trained_model, train_examples)
            te_c, te_t = evaluate(trained_model, test_examples)
            print(f"  step {step:5d} | loss {loss.item():.4f} | train {tr_c}/{tr_t} | TEST {te_c}/{te_t}")

    tr_c, tr_t = evaluate(trained_model, train_examples)
    te_c, te_t = evaluate(trained_model, test_examples)
    print("\n" + "=" * 60)
    print("  FINAL: Trained Model")
    print("=" * 60)
    print(f"  Seen (train):   {tr_c}/{tr_t} ({100*tr_c/tr_t:.1f}%)")
    print(f"  Unseen (test):  {te_c}/{te_t} ({100*te_c/te_t:.1f}%)")
