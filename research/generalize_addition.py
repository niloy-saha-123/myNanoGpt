"""
Generalization Test: Decimal Addition

This is the most interesting test. We scale up to DECIMAL (base-10) addition:
  a + b = (tens digit)(ones digit)
  e.g. 7 + 8 = 15

There are 100 possible inputs (0+0 through 9+9).
We train on 50 and test on 50 the model has NEVER seen.

The hand-designed model always gets 100/100 because it encodes the algorithm.
Can SGD learn the actual addition rule, or does it just memorize?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import random
from model import GPT

# ---------- config ----------
# vocab: 0-9 (digits) + 10 (+) + 11 (=)
vocab_size = 12
n_embd     = 25
n_head     = 1
n_layer    = 1
block_size = 6   # a + b = d1 d0
device     = 'cpu'

itos = {i: str(i) for i in range(10)}
itos[10] = '+'
itos[11] = '='
stoi = {v: k for k, v in itos.items()}

lr         = 1e-3
max_iters  = 20000
eval_every = 2000

# ---------- build all 100 examples ----------
all_examples = []
for a in range(10):
    for b in range(10):
        all_examples.append((a, b))

# ---------- split ----------
random.seed(42)
random.shuffle(all_examples)
train_examples = all_examples[:50]
test_examples  = all_examples[50:]

def make_dataset(examples):
    inputs, targets = [], []
    for (a, b) in examples:
        s = a + b
        d1 = s // 10  # tens digit
        d0 = s % 10   # ones digit
        # sequence: [a, +, b, =, d1, d0]
        seq = [a, stoi['+'], b, stoi['='], d1, d0]
        inp = seq[:-1]  # [a, +, b, =, d1]
        tgt = seq[1:]   # [+, b, =, d1, d0]
        # only care about output positions (predict d1 and d0)
        masked_tgt = [-100, -100, -100, tgt[3], tgt[4]]
        inputs.append(inp)
        targets.append(masked_tgt)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def evaluate(model, examples):
    model.eval()
    correct, total = 0, 0
    results = []
    with torch.no_grad():
        for (a, b) in examples:
            s = a + b
            expected_d1 = s // 10
            expected_d0 = s % 10
            context = [a, stoi['+'], b, stoi['=']]
            idx = torch.tensor([context], device=device)
            preds = []
            for _ in range(2):  # predict 2 output tokens
                logits, _ = model(idx)
                next_tok = logits[0, -1, :].argmax().item()
                idx = torch.cat([idx, torch.tensor([[next_tok]], device=device)], dim=1)
                preds.append(next_tok)
            ok = preds == [expected_d1, expected_d0]
            if ok: correct += 1
            total += 1
            results.append((a, b, preds, [expected_d1, expected_d0], ok))
    model.train()
    return correct, total, results


def build_hand_addition():
    """
    Build the hand-designed decimal addition model.
    Replicates hand_decimal_addition_ar.py: sets embeddings + attention by hand,
    then calibrates lm_head with a small optimization loop.
    """
    BASE = 10
    m = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    with torch.no_grad():
        tok_emb = torch.zeros(vocab_size, n_embd)
        for i in range(vocab_size): tok_emb[i, i] = 1.0
        m.token_embedding_table.weight.copy_(tok_emb)

        pos_emb = torch.zeros(block_size, n_embd)
        pos_emb[0, 12] = 1.0   # operand a
        pos_emb[2, 13] = 1.0   # operand b
        pos_emb[3, 14] = 1.0   # output position
        pos_emb[4, 14] = 1.0   # output position
        m.position_embedding_table.weight.copy_(pos_emb)

        head = m.blocks[0].sa.heads[0]
        W_Q = torch.zeros(n_embd, n_embd)
        W_Q[12, 14] = 10.0; W_Q[13, 14] = 10.0
        head.query.weight.copy_(W_Q)
        W_K = torch.zeros(n_embd, n_embd)
        W_K[12, 12] = 10.0; W_K[13, 13] = 10.0
        head.key.weight.copy_(W_K)
        W_V = torch.zeros(n_embd, n_embd)
        for d in range(BASE): W_V[15 + d, d] = 1.0
        head.value.weight.copy_(W_V)
        W_proj = torch.zeros(n_embd, n_embd)
        for d in range(BASE): W_proj[15 + d, 15 + d] = 1.0
        m.blocks[0].sa.proj.weight.copy_(W_proj)
        m.blocks[0].sa.proj.bias.zero_()

        m.blocks[0].ffwd.net[0].weight.zero_(); m.blocks[0].ffwd.net[0].bias.zero_()
        m.blocks[0].ffwd.net[2].weight.zero_(); m.blocks[0].ffwd.net[2].bias.zero_()
        m.blocks[0].ln1.weight.fill_(1.0); m.blocks[0].ln1.bias.zero_()
        m.blocks[0].ln2.weight.fill_(1.0); m.blocks[0].ln2.bias.zero_()
        m.ln_f.weight.fill_(1.0); m.ln_f.bias.zero_()

        # Calibrate lm_head from post-LN_f vectors for all 200 cases
        m.eval()
        vectors, tgts = [], []
        for a in range(BASE):
            for b in range(BASE):
                s = a + b
                d1, d0 = s // BASE, s % BASE
                ctx1 = [a, stoi['+'], b, stoi['=']]
                idx1 = torch.tensor([ctx1], device=device)
                tok = m.token_embedding_table(idx1)
                pos = m.position_embedding_table(torch.arange(len(ctx1), device=device))
                x = tok + pos; x = m.blocks(x); x = m.ln_f(x)
                vectors.append(x[0, -1].clone()); tgts.append(d1)
                ctx2 = [a, stoi['+'], b, stoi['='], d1]
                idx2 = torch.tensor([ctx2], device=device)
                tok = m.token_embedding_table(idx2)
                pos = m.position_embedding_table(torch.arange(len(ctx2), device=device))
                x = tok + pos; x = m.blocks(x); x = m.ln_f(x)
                vectors.append(x[0, -1].clone()); tgts.append(d0)

        X = torch.stack(vectors)
        target_tensor = torch.tensor(tgts, device=device)
        W = torch.zeros(vocab_size, n_embd)
        b = torch.zeros(vocab_size)
        for step in range(20000):
            logits = X @ W.T + b
            if (logits.argmax(dim=1) == target_tensor).all():
                break
            loss = F.cross_entropy(logits, target_tensor)
            probs = F.softmax(logits, dim=1)
            grad = probs.clone()
            for i in range(len(tgts)):
                grad[i, tgts[i]] -= 1.0
            grad /= len(tgts)
            W -= 1.0 * (grad.T @ X)
            b -= 1.0 * grad.sum(dim=0)
        m.lm_head.weight.copy_(W)
        m.lm_head.bias.copy_(b)
    return m


if __name__ == '__main__':
    torch.manual_seed(42)

    print("=" * 60)
    print("  GENERALIZATION TEST: Decimal Addition")
    print("=" * 60)
    print(f"\n  Train set: {len(train_examples)} examples")
    print(f"  Test set:  {len(test_examples)} examples (NEVER seen during training)")

    train_sums = sorted([f"{a}+{b}={a+b}" for a, b in train_examples])
    test_sums  = sorted([f"{a}+{b}={a+b}" for a, b in test_examples])
    print(f"  Some train: {train_sums[:10]}...")
    print(f"  Some test:  {test_sums[:10]}...")

    # --- Build hand-designed model ---
    print("\nBuilding hand-designed model (calibrating lm_head)...")
    hand_model = build_hand_addition()

    # --- Train SGD model ---
    trained_model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    optimizer = torch.optim.AdamW(trained_model.parameters(), lr=lr)
    train_inputs, train_targets = make_dataset(train_examples)
    train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)

    print(f"Training SGD model on {len(train_examples)} examples...\n")
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
    print("  4-WAY COMPARISON: Decimal Addition")
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
        
        # Print side-by-side in columns
        for i in range(0, total, 4):
            cols = []
            for (a, b, preds, expected, ok) in results[i:i+4]:
                pred_str = ''.join(itos[p] for p in preds)
                exp_str  = ''.join(itos[e] for e in expected)
                mark = "PASS" if ok else "FAIL"
                # Shorter format to fit 4 cols: 7+8=15 (exp 15) PASS
                res_str = f"{a}+{b}={pred_str} (e:{exp_str}) {mark}"
                cols.append(f"{res_str:<22s}")
            print("    " + " | ".join(cols))
