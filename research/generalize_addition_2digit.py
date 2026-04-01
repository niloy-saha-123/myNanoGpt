"""
Generalization Test: 2-Digit Decimal Addition (100 × 100 = 10,000 pairs)

Two 2-digit numbers: ab + cd = (up to 3 digits, max 99+99=198)
e.g. 45 + 67 = 112

Input space: 100 × 100 = 10,000 possible pairs (0-99 + 0-99).
Compare to 1-digit addition (100 pairs) and 3-digit (1M pairs).

Usage:
  python generalize_addition_2digit.py           # default 2000 train / 2000 test
  python generalize_addition_2digit.py 5000     # 5k train
  python generalize_addition_2digit.py 8000      # 8k train (most of the space)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import os
import random
from model import GPT

vocab_size = 12   # 0-9, +, =
n_embd     = 48   # larger for 2-digit
n_head     = 2
n_layer    = 1
# seq: ab + cd = xyz  -> 2+1+2+1+3 = 9 tokens (sum max 198 = 3 digits)
block_size = 9
device     = 'cpu' if os.environ.get('RESEARCH_USE_CPU') else ('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

itos = {i: str(i) for i in range(10)}
itos[10] = '+'
itos[11] = '='
stoi = {v: k for k, v in itos.items()}

lr         = 1e-3
max_iters  = 25000
eval_every = 2500

TRAIN_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
TEST_SIZE  = 2000
TOTAL_PAIRS = 100 * 100

random.seed(42)
all_pairs = [(a, b) for a in range(100) for b in range(100)]
random.shuffle(all_pairs)
train_examples = all_pairs[:TRAIN_SIZE]
test_examples  = all_pairs[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]


def to_digits(n, pad=2):
    """Convert int to list of digits, padded."""
    if n == 0:
        return [0] * pad
    digs = []
    while n:
        digs.append(n % 10)
        n //= 10
    digs.reverse()
    return [0] * (pad - len(digs)) + digs


def add_2digit(a, b):
    """a, b in 0..99. Return 3 digits: hundreds, tens, ones."""
    s = a + b  # max 198
    d2 = (s // 100) % 10
    d1 = (s // 10) % 10
    d0 = s % 10
    return [d2, d1, d0]


def make_dataset(examples):
    inputs, targets = [], []
    for (a, b) in examples:
        a_d = to_digits(a, 2)
        b_d = to_digits(b, 2)
        out = add_2digit(a, b)
        seq = a_d + [stoi['+']] + b_d + [stoi['=']] + out
        inp = seq[:-1]
        tgt = seq[1:]
        masked_tgt = [-100] * len(tgt)
        for i in [5, 6, 7]:  # output digits d2,d1,d0
            masked_tgt[i] = tgt[i]
        inputs.append(inp)
        targets.append(masked_tgt)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def evaluate(model, examples):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (a, b) in examples:
            a_d = to_digits(a, 2)
            b_d = to_digits(b, 2)
            expected = add_2digit(a, b)
            context = a_d + [stoi['+']] + b_d + [stoi['=']]
            idx = torch.tensor([context], device=device)
            preds = []
            for _ in range(3):
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
    print("  GENERALIZATION TEST: 2-Digit Decimal Addition")
    print("=" * 60)
    print(f"\n  Format: ab + cd = xyz (e.g. 45 + 67 = 112)")
    print(f"  Input space: 100 × 100 = {TOTAL_PAIRS:,} possible pairs")
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
    print(f"  Unseen (test): {te_c}/{te_t} ({100*te_c/te_t:.1f}%)")
