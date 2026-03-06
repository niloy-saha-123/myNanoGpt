"""
Generalization Test: Copy Task

Train on a SUBSET of the 27 possible inputs, then test on HELD-OUT inputs
the model has never seen. Compare to the hand-designed model which always
generalizes because it encodes the actual algorithm.

Split: 18 train / 9 test (randomly selected)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import random
from model import GPT

# ---------- config (same architecture as hand_copy_ar.py) ----------
vocab_size = 4
n_embd     = 11
n_head     = 1
n_layer    = 1
block_size = 7
device     = 'cpu'

itos = {0: 'A', 1: 'B', 2: 'C', 3: '<sep>'}
stoi = {'A': 0, 'B': 1, 'C': 2, '<sep>': 3}

lr         = 1e-3
max_iters  = 5000
eval_every = 500

# ---------- build all 27 examples ----------
all_examples = []
for a in range(3):
    for b in range(3):
        for c in range(3):
            all_examples.append((a, b, c))

# ---------- split into train/test ----------
random.seed(42)
random.shuffle(all_examples)
train_examples = all_examples[:9]
test_examples  = all_examples[9:]

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

def evaluate(model, examples, label=""):
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
            if ok: correct += 1
            total += 1
            results.append((a, b, c, preds, ok))
    model.train()
    return correct, total, results


def build_hand_copy():
    """Build the hand-designed copy model (same logic as hand_copy_ar.py)."""
    m = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    with torch.no_grad():
        tok_emb = torch.zeros(vocab_size, n_embd)
        for i in range(vocab_size): tok_emb[i, i] = 1.0
        m.token_embedding_table.weight.copy_(tok_emb)
        pos_emb = torch.zeros(block_size, n_embd)
        for i in range(block_size): pos_emb[i, vocab_size + i] = 1.0
        m.position_embedding_table.weight.copy_(pos_emb)
        head = m.blocks[0].sa.heads[0]
        W_Q = torch.zeros(n_embd, n_embd)
        for t in range(block_size): W_Q[vocab_size + t, vocab_size + t] = 100.0
        head.query.weight.copy_(W_Q)
        W_K = torch.zeros(n_embd, n_embd)
        for t in range(3, block_size): W_K[vocab_size + t, vocab_size + t - 3] = 100.0
        head.key.weight.copy_(W_K)
        W_V = torch.zeros(n_embd, n_embd)
        for i in range(vocab_size): W_V[i, i] = 1.0
        head.value.weight.copy_(W_V)
        W_proj = torch.zeros(n_embd, n_embd)
        for i in range(vocab_size): W_proj[i, i] = 1.0
        m.blocks[0].sa.proj.weight.copy_(W_proj)
        m.blocks[0].sa.proj.bias.zero_()
        m.blocks[0].ffwd.net[0].weight.zero_(); m.blocks[0].ffwd.net[0].bias.zero_()
        m.blocks[0].ffwd.net[2].weight.zero_(); m.blocks[0].ffwd.net[2].bias.zero_()
        m.blocks[0].ln1.weight.fill_(1.0); m.blocks[0].ln1.bias.zero_()
        m.blocks[0].ln2.weight.fill_(1.0); m.blocks[0].ln2.bias.zero_()
        m.ln_f.weight.fill_(1.0); m.ln_f.bias.zero_()
        W_out = torch.zeros(vocab_size, n_embd)
        for i in range(vocab_size): W_out[i, i] = 1.0
        m.lm_head.weight.copy_(W_out); m.lm_head.bias.zero_()
    return m


if __name__ == '__main__':
    torch.manual_seed(42)

    print("=" * 60)
    print("  GENERALIZATION TEST: Copy Task")
    print("=" * 60)
    print(f"\n  Train set: {len(train_examples)} examples")
    print(f"  Test set:  {len(test_examples)} examples (NEVER seen during training)")
    print(f"  Train: {[''.join(itos[x] for x in t) for t in train_examples]}")
    print(f"  Test:  {[''.join(itos[x] for x in t) for t in test_examples]}")

    # --- Build hand-designed model ---
    hand_model = build_hand_copy()

    # --- Train SGD model ---
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

    # === 4-WAY COMPARISON ===
    print("\n" + "=" * 60)
    print("  4-WAY COMPARISON: Copy Task")
    print("=" * 60)

    h_seen_c,  h_seen_t,  h_seen_r  = evaluate(hand_model,    train_examples)
    h_unseen_c, h_unseen_t, h_unseen_r = evaluate(hand_model, test_examples)
    t_seen_c,  t_seen_t,  t_seen_r  = evaluate(trained_model, train_examples)
    t_unseen_c, t_unseen_t, t_unseen_r = evaluate(trained_model, test_examples)

    print(f"\n  {'':30s} | {'Seen (train)':>14s} | {'Unseen (test)':>14s}")
    print(f"  {'-'*30}-+-{'-'*14}-+-{'-'*14}")
    print(f"  {'Hand-Designed Model':30s} | {h_seen_c:>5d}/{h_seen_t:<5d}    | {h_unseen_c:>5d}/{h_unseen_t:<5d}")
    print(f"  {'Trained (SGD) Model':30s} | {t_seen_c:>5d}/{t_seen_t:<5d}    | {t_unseen_c:>5d}/{t_unseen_t:<5d}")

    # --- Detailed results for each quadrant ---
    for label, results in [("HAND-DESIGNED on SEEN", h_seen_r),
                           ("HAND-DESIGNED on UNSEEN", h_unseen_r),
                           ("TRAINED on SEEN", t_seen_r),
                           ("TRAINED on UNSEEN", t_unseen_r)]:
        correct = sum(1 for r in results if r[4])
        total = len(results)
        print(f"\n  {label} ({correct}/{total}):")
        for (a, b, c, preds, ok) in results:
            in_str = ''.join(itos[x] for x in [a, b, c])
            out_str = ''.join(itos[x] for x in preds)
            mark = "PASS" if ok else "FAIL"
            print(f"    {in_str} -> {out_str}  {mark}")
