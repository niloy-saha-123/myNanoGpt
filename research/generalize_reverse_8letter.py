"""
Generalization Test: Reverse Task (8 letters)

Same input/output format as base reverse: INPUT<sep> -> OUTPUT (reversed)
Now 8 letters (A..H), 8 positions. Input space: 8^8 = 16,777,216.

Format: ABCDEFGH<sep> -> HGFEDCBA (matches base "ABC<sep> -> CBA" style)

Train on a SUBSET, test on HELD-OUT. Hand-designed vs trained.
Configurable train size for more/less data comparison.

Usage:
  python generalize_reverse_8letter.py           # default 1000 train / 2000 test
  python generalize_reverse_8letter.py 5000      # 5k train / 2k test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import os
import random
from model import GPT

# ---------- config (8 letters A..H, same style as base) ----------
NUM_LETTERS = 8
SEQ_LEN = NUM_LETTERS
vocab_size = NUM_LETTERS + 1   # A..H + <sep>
block_size = SEQ_LEN + 1 + SEQ_LEN   # 17
n_embd     = vocab_size + block_size
n_head     = 1
n_layer    = 1
device     = 'cpu' if os.environ.get('RESEARCH_USE_CPU') else ('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

itos = {i: chr(ord('A') + i) for i in range(NUM_LETTERS)}
itos[NUM_LETTERS] = '<sep>'
stoi = {v: k for k, v in itos.items()}

lr         = 1e-3
max_iters  = 15000
eval_every = 1500

# ---------- configurable split ----------
TRAIN_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
TEST_SIZE  = 2000
random.seed(42)
all_examples = []
for _ in range(TRAIN_SIZE + TEST_SIZE):
    all_examples.append(tuple(random.randint(0, NUM_LETTERS - 1) for _ in range(SEQ_LEN)))
train_examples = all_examples[:TRAIN_SIZE]
test_examples  = all_examples[TRAIN_SIZE:]


def make_dataset(examples):
    inputs, targets = [], []
    for tup in examples:
        rev = list(reversed(tup))
        seq = list(tup) + [stoi['<sep>']] + rev
        inp = seq[:-1]
        tgt = seq[1:]
        masked_tgt = [-100] * len(tgt)
        for i in range(SEQ_LEN, len(tgt)):
            masked_tgt[i] = tgt[i]
        inputs.append(inp)
        targets.append(masked_tgt)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def evaluate(model, examples):
    model.eval()
    correct, total = 0, 0
    results = []
    with torch.no_grad():
        for tup in examples:
            expected = list(reversed(tup))
            context = list(tup) + [stoi['<sep>']]
            idx = torch.tensor([context], device=device)
            preds = []
            for _ in range(SEQ_LEN):
                logits, _ = model(idx)
                next_tok = logits[0, -1, :].argmax().item()
                idx = torch.cat([idx, torch.tensor([[next_tok]], device=device)], dim=1)
                preds.append(next_tok)
            ok = preds == expected
            if ok:
                correct += 1
            total += 1
            results.append((tup, preds, expected, ok))
    model.train()
    return correct, total, results


def build_hand_reverse():
    """Hand-designed reverse: at output pos t, attend to pos (16-t)."""
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
        W_K = torch.zeros(n_embd, n_embd)
        # Output pos t (9..16) -> attend to (block_size-1)-t = 16-t
        for t in range(SEQ_LEN + 1, block_size):
            W_K[vocab_size + t, vocab_size + (block_size - 1 - t)] = 100.0
        head.key.weight.copy_(W_K)
        head.query.weight.copy_(W_Q)
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
    all_examples = []
    for _ in range(TRAIN_SIZE + TEST_SIZE):
        all_examples.append(tuple(random.randint(0, NUM_LETTERS - 1) for _ in range(SEQ_LEN)))
    train_examples = all_examples[:TRAIN_SIZE]
    test_examples = all_examples[TRAIN_SIZE:]

    print("=" * 60)
    print("  GENERALIZATION TEST: Reverse Task (8 letters A..H)")
    print("=" * 60)
    print(f"\n  Format: INPUT<sep> -> OUTPUT reversed (same as base)")
    print(f"  Input space: {NUM_LETTERS}^{SEQ_LEN} = {NUM_LETTERS**SEQ_LEN:,}")
    print(f"  Train set: {len(train_examples):,} examples")
    print(f"  Test set:  {len(test_examples):,} examples (NEVER seen)")

    hand_model = build_hand_reverse()
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
            tr_c, tr_t, _ = evaluate(trained_model, train_examples)
            te_c, te_t, _ = evaluate(trained_model, test_examples)
            print(f"  step {step:5d} | loss {loss.item():.4f} | train {tr_c}/{tr_t} | TEST {te_c}/{te_t}")

    print("\n" + "=" * 60)
    print("  4-WAY COMPARISON: Reverse 8-Letter")
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
        correct = sum(1 for r in results if r[3])
        total = len(results)
        print(f"\n  {label} ({correct}/{total}):")
        for (tup, preds, expected, ok) in results[:8]:
            in_str = ''.join(itos[x] for x in tup)
            out_str = ''.join(itos[x] for x in preds)
            exp_str = ''.join(itos[x] for x in expected)
            mark = "PASS" if ok else "FAIL"
            print(f"    {in_str} -> {out_str} (expected {exp_str})  {mark}")
        if total > 8:
            print(f"    ... and {total - 8} more")
