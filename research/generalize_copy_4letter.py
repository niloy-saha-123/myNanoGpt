"""
Generalization Test: Copy Task (4 letters)

Same as generalize_copy but with 4 letters (A,B,C,D) instead of 3.
Input space: 4^3 = 64 possible sequences.

Train on a SUBSET, test on HELD-OUT. Compare hand-designed vs trained.
Use different train/test splits to study effect of more training data.

Usage:
  python generalize_copy_4letter.py           # default TRAIN_SIZE=32
  python generalize_copy_4letter.py 16       # 16 train / 48 test
  python generalize_copy_4letter.py 48        # 48 train / 16 test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import os
import random
from model import GPT

# ---------- config (4 letters: A,B,C,D) ----------
NUM_LETTERS = 4
vocab_size = NUM_LETTERS + 1   # A,B,C,D + <sep>
n_embd     = vocab_size + 7    # token dims + position dims for seq len 7
n_head     = 1
n_layer    = 1
block_size = 7                 # ABC<sep>ABC or ABCD<sep>ABCD... wait, 3 positions
# For 3 positions with 4 letters: seq = [a,b,c,sep,a,b,c], length 7
device     = 'cpu' if os.environ.get('RESEARCH_USE_CPU') else ('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

itos = {i: chr(ord('A')+i) for i in range(NUM_LETTERS)}
itos[NUM_LETTERS] = '<sep>'
stoi = {v: k for k, v in itos.items()}

lr         = 1e-3
max_iters  = 8000
eval_every = 1000

# ---------- build all 64 examples (4^3) ----------
all_examples = []
for a in range(NUM_LETTERS):
    for b in range(NUM_LETTERS):
        for c in range(NUM_LETTERS):
            all_examples.append((a, b, c))

# ---------- split (configurable via cmd line: 16, 32, 48) ----------
random.seed(42)
random.shuffle(all_examples)
TRAIN_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 32
train_examples = all_examples[:TRAIN_SIZE]
test_examples  = all_examples[TRAIN_SIZE:]


def make_dataset(examples):
    inputs, targets = [], []
    for (a, b, c) in examples:
        seq = [a, b, c, stoi['<sep>'], a, b, c]
        inp = seq[:-1]
        tgt = seq[1:]
        masked_tgt = [-100, -100, -100, tgt[3], tgt[4], tgt[5]]
        inputs.append(inp)
        targets.append(masked_tgt)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def evaluate(model, examples):
    model.eval()
    correct, total = 0, 0
    results = []
    with torch.no_grad():
        for (a, b, c) in examples:
            context = [a, b, c, stoi['<sep>']]
            idx = torch.tensor([context], device=device)
            preds = []
            for _ in range(3):
                logits, _ = model(idx)
                next_tok = logits[0, -1, :].argmax().item()
                idx = torch.cat([idx, torch.tensor([[next_tok]], device=device)], dim=1)
                preds.append(next_tok)
            ok = preds == [a, b, c]
            if ok:
                correct += 1
            total += 1
            results.append((a, b, c, preds, ok))
    model.train()
    return correct, total, results


def build_hand_copy():
    """Hand-designed copy: attention looks back 3 positions, copies token."""
    m = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    with torch.no_grad():
        tok_emb = torch.zeros(vocab_size, n_embd)
        for i in range(vocab_size):
            tok_emb[i, i] = 1.0
        m.token_embedding_table.weight.copy_(tok_emb)
        pos_emb = torch.zeros(block_size, n_embd)
        for i in range(block_size):
            pos_emb[i, vocab_size + i] = 1.0
        m.position_embedding_table.weight.copy_(pos_emb)
        head = m.blocks[0].sa.heads[0]
        W_Q = torch.zeros(n_embd, n_embd)
        for t in range(block_size):
            W_Q[vocab_size + t, vocab_size + t] = 100.0
        head.query.weight.copy_(W_Q)
        W_K = torch.zeros(n_embd, n_embd)
        for t in range(3, block_size):
            W_K[vocab_size + t, vocab_size + t - 3] = 100.0
        head.key.weight.copy_(W_K)
        W_V = torch.zeros(n_embd, n_embd)
        for i in range(vocab_size):
            W_V[i, i] = 1.0
        head.value.weight.copy_(W_V)
        W_proj = torch.zeros(n_embd, n_embd)
        for i in range(vocab_size):
            W_proj[i, i] = 1.0
        m.blocks[0].sa.proj.weight.copy_(W_proj)
        m.blocks[0].sa.proj.bias.zero_()
        m.blocks[0].ffwd.net[0].weight.zero_()
        m.blocks[0].ffwd.net[0].bias.zero_()
        m.blocks[0].ffwd.net[2].weight.zero_()
        m.blocks[0].ffwd.net[2].bias.zero_()
        m.blocks[0].ln1.weight.fill_(1.0)
        m.blocks[0].ln1.bias.zero_()
        m.blocks[0].ln2.weight.fill_(1.0)
        m.blocks[0].ln2.bias.zero_()
        m.ln_f.weight.fill_(1.0)
        m.ln_f.bias.zero_()
        W_out = torch.zeros(vocab_size, n_embd)
        for i in range(vocab_size):
            W_out[i, i] = 1.0
        m.lm_head.weight.copy_(W_out)
        m.lm_head.bias.zero_()
    return m


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 60)
    print("  GENERALIZATION TEST: Copy Task (4 letters A,B,C,D)")
    print("=" * 60)
    print(f"\n  Total possible: {len(all_examples)}")
    print(f"  Train set: {len(train_examples)} examples")
    print(f"  Test set:  {len(test_examples)} examples (NEVER seen during training)")

    hand_model = build_hand_copy()
    trained_model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    optimizer = torch.optim.AdamW(trained_model.parameters(), lr=lr)
    train_inputs, train_targets = make_dataset(train_examples)
    train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)

    print(f"\nTraining on {len(train_examples)} examples...\n")
    for step in range(max_iters + 1):
        logits, loss = trained_model(train_inputs, train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            tr_c, tr_t, _ = evaluate(trained_model, train_examples)
            te_c, te_t, _ = evaluate(trained_model, test_examples)
            print(f"  step {step:5d} | loss {loss.item():.4f} | train {tr_c}/{tr_t} | TEST {te_c}/{te_t}")

    print("\n" + "=" * 60)
    print("  4-WAY COMPARISON: Copy 4-Letter")
    print("=" * 60)

    h_seen_c, h_seen_t, h_seen_r = evaluate(hand_model, train_examples)
    h_unseen_c, h_unseen_t, h_unseen_r = evaluate(hand_model, test_examples)
    t_seen_c, t_seen_t, t_seen_r = evaluate(trained_model, train_examples)
    t_unseen_c, t_unseen_t, t_unseen_r = evaluate(trained_model, test_examples)

    print(f"\n  {'':30s} | {'Seen (train)':>14s} | {'Unseen (test)':>14s}")
    print(f"  {'-'*30}-+-{'-'*14}-+-{'-'*14}")
    print(f"  {'Hand-Designed Model':30s} | {h_seen_c:>5d}/{h_seen_t:<5d}    | {h_unseen_c:>5d}/{h_unseen_t:<5d}")
    print(f"  {'Trained (SGD) Model':30s} | {t_seen_c:>5d}/{t_seen_t:<5d}    | {t_unseen_c:>5d}/{t_unseen_t:<5d}")

    for label, results in [("HAND-DESIGNED on SEEN", h_seen_r),
                           ("HAND-DESIGNED on UNSEEN", h_unseen_r),
                           ("TRAINED on SEEN", t_seen_r),
                           ("TRAINED on UNSEEN", t_unseen_r)]:
        correct = sum(1 for r in results if r[4])
        total = len(results)
        print(f"\n  {label} ({correct}/{total}):")
        for (a, b, c, preds, ok) in results[:10]:  # show first 10
            in_str = ''.join(itos[x] for x in [a, b, c])
            out_str = ''.join(itos[x] for x in preds)
            mark = "PASS" if ok else "FAIL"
            print(f"    {in_str} -> {out_str}  {mark}")
        if total > 10:
            print(f"    ... and {total - 10} more")
