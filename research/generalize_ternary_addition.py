"""
Generalization Test: 1-Digit Ternary Addition

Compare a hand-designed autoregressive ternary adder to an SGD-trained GPT.
Inputs:  a + b =
Outputs: two base-3 digits d1 d0 (carry, ones). Total 9 input pairs.
We train on a subset, test on held-out pairs, and report a 4-way table.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import random
from model import GPT

# ---------- config ----------
vocab_size = 5           # 0,1,2,+,=
n_embd     = 12
n_head     = 1
n_layer    = 1
block_size = 6           # a + b = d1 d0
device     = 'mps' if torch.backends.mps.is_available() else 'cpu'
BASE       = 3

itos = {0: '0', 1: '1', 2: '2', 3: '+', 4: '='}
stoi = {v: k for k, v in itos.items()}

lr         = 1e-3
max_iters  = 8000
eval_every = 1000

# ---------- build all 9 examples ----------
all_examples = [(a, b) for a in range(BASE) for b in range(BASE)]

# ---------- split (train 5, test 4) ----------
random.seed(42)
random.shuffle(all_examples)
train_examples = all_examples[:5]
test_examples  = all_examples[5:]

def make_dataset(examples):
    inputs, targets = [], []
    for (a, b) in examples:
        s = a + b
        d1 = s // BASE
        d0 = s % BASE
        seq = [a, stoi['+'], b, stoi['='], d1, d0]
        inp = seq[:-1]
        tgt = seq[1:]
        masked_tgt = [-100, -100, -100, tgt[3], tgt[4]]
        inputs.append(inp)
        targets.append(masked_tgt)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def evaluate(model, examples):
    model.eval()
    correct, total = 0, 0
    rows = []
    with torch.no_grad():
        for (a, b) in examples:
            s = a + b
            expected = [s // BASE, s % BASE]
            ctx = [a, stoi['+'], b, stoi['=']]
            idx = torch.tensor([ctx], device=device)
            preds = []
            for _ in range(2):
                logits, _ = model(idx)
                next_tok = logits[0, -1, :].argmax().item()
                idx = torch.cat([idx, torch.tensor([[next_tok]], device=device)], dim=1)
                preds.append(next_tok)
            ok = preds == expected
            correct += int(ok)
            total += 1
            rows.append((a, b, preds, expected, ok))
    model.train()
    return correct, total, rows


def build_hand_ternary():
    """Replicates hand_ternary_addition_ar with lm_head calibration."""
    m = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    with torch.no_grad():
        tok_emb = torch.zeros(vocab_size, n_embd)
        for i in range(vocab_size):
            tok_emb[i, i] = 1.0
        m.token_embedding_table.weight.copy_(tok_emb)

        pos_emb = torch.zeros(block_size, n_embd)
        pos_emb[0, 5] = 1.0  # operand a
        pos_emb[2, 6] = 1.0  # operand b
        pos_emb[3, 7] = 1.0  # output pos (d1)
        pos_emb[4, 7] = 1.0  # output pos (d0)
        m.position_embedding_table.weight.copy_(pos_emb)

        head = m.blocks[0].sa.heads[0]
        W_Q = torch.zeros(n_embd, n_embd)
        W_Q[5, 7] = 10.0
        W_Q[6, 7] = 10.0
        head.query.weight.copy_(W_Q)

        W_K = torch.zeros(n_embd, n_embd)
        W_K[5, 5] = 10.0
        W_K[6, 6] = 10.0
        head.key.weight.copy_(W_K)

        W_V = torch.zeros(n_embd, n_embd)
        for d in range(BASE):
            W_V[8 + d, d] = 1.0
        head.value.weight.copy_(W_V)

        W_proj = torch.zeros(n_embd, n_embd)
        for d in range(BASE):
            W_proj[8 + d, 8 + d] = 1.0
        m.blocks[0].sa.proj.weight.copy_(W_proj)
        m.blocks[0].sa.proj.bias.zero_()

        m.blocks[0].ffwd.net[0].weight.zero_(); m.blocks[0].ffwd.net[0].bias.zero_()
        m.blocks[0].ffwd.net[2].weight.zero_(); m.blocks[0].ffwd.net[2].bias.zero_()
        m.blocks[0].ln1.weight.fill_(1.0); m.blocks[0].ln1.bias.zero_()
        m.blocks[0].ln2.weight.fill_(1.0); m.blocks[0].ln2.bias.zero_()
        m.ln_f.weight.fill_(1.0); m.ln_f.bias.zero_()

        # Calibrate lm_head using post-LN vectors for all (a,b,step)
        m.eval()
        vectors, targets = [], []
        for a in range(BASE):
            for b in range(BASE):
                s = a + b
                d1, d0 = s // BASE, s % BASE

                ctx1 = [a, stoi['+'], b, stoi['=']]
                idx1 = torch.tensor([ctx1], device=device)
                tok = m.token_embedding_table(idx1)
                pos = m.position_embedding_table(torch.arange(len(ctx1), device=device))
                x = tok + pos; x = m.blocks(x); x = m.ln_f(x)
                vectors.append(x[0, -1].clone()); targets.append(d1)

                ctx2 = [a, stoi['+'], b, stoi['='], d1]
                idx2 = torch.tensor([ctx2], device=device)
                tok = m.token_embedding_table(idx2)
                pos = m.position_embedding_table(torch.arange(len(ctx2), device=device))
                x = tok + pos; x = m.blocks(x); x = m.ln_f(x)
                vectors.append(x[0, -1].clone()); targets.append(d0)

        X = torch.stack(vectors)
        target_tensor = torch.tensor(targets, device=device)
        W = torch.zeros(vocab_size, n_embd, device=device)
        b = torch.zeros(vocab_size, device=device)
        lr_cal = 1.0
        for _ in range(2000):
            logits = X @ W.T + b
            if (logits.argmax(dim=1) == target_tensor).all():
                break
            probs = F.softmax(logits, dim=1)
            grad = probs.clone()
            for i in range(len(targets)):
                grad[i, targets[i]] -= 1.0
            grad /= len(targets)
            W -= lr_cal * (grad.T @ X)
            b -= lr_cal * grad.sum(dim=0)
        m.lm_head.weight.copy_(W)
        m.lm_head.bias.copy_(b)
    return m


if __name__ == '__main__':
    torch.manual_seed(42)

    print("=" * 60)
    print("  GENERALIZATION TEST: Ternary Addition")
    print("=" * 60)
    print(f"\n  Train set: {len(train_examples)} examples")
    print(f"  Test set:  {len(test_examples)} examples (NEVER seen during training)")

    hand_model = build_hand_ternary()

    trained_model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    optimizer = torch.optim.AdamW(trained_model.parameters(), lr=lr)
    train_inputs, train_targets = make_dataset(train_examples)
    train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)

    print(f"\nTraining SGD model on {len(train_examples)} examples...\n")
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
    print("  4-WAY COMPARISON: Ternary Addition")
    print("=" * 60)

    h_seen_c,  h_seen_t,  h_seen_r  = evaluate(hand_model,    train_examples)
    h_unseen_c, h_unseen_t, h_unseen_r = evaluate(hand_model, test_examples)
    t_seen_c,  t_seen_t,  t_seen_r  = evaluate(trained_model, train_examples)
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
        for (a, b, preds, expected, ok) in results:
            pred_str = ''.join(itos[p] for p in preds)
            exp_str  = ''.join(itos[e] for e in expected)
            mark = "PASS" if ok else "FAIL"
            print(f"    {a}+{b} -> {pred_str} (expected {exp_str}) {mark}")
