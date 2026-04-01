"""
Generalization Test: 2-Digit Ternary Addition (base-3)

Inputs:  a1 a0 + b1 b0 =
Outputs: d2 d1 d0 (three base-3 digits to cover max sum). There are 81 input
pairs (9x9). We train on a subset, test on held-out pairs, and compare a
hand-designed model to an SGD-trained GPT.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import random
from model import GPT

# ---------- config ----------
BASE       = 3
vocab_size = 6              # 0,1,2,+,=,<pad?> (we use 0-2,+,=; 5 unused)
n_embd     = 24
n_head     = 4
n_layer    = 1
block_size = 9              # a1 a0 + b1 b0 = d2 d1 d0
device     = 'mps' if torch.backends.mps.is_available() else 'cpu'

itos = {0: '0', 1: '1', 2: '2', 3: '+', 4: '=', 5: '<pad>'}
stoi = {v: k for k, v in itos.items()}

lr         = 1e-3
max_iters  = 20000
eval_every = 2000

# ---------- build all 81 examples ----------
all_examples = []
for a1 in range(BASE):
    for a0 in range(BASE):
        for b1 in range(BASE):
            for b0 in range(BASE):
                all_examples.append((a1, a0, b1, b0))

# ---------- split (configurable: 70 default, 80 = max data for 81 pairs) ----------
random.seed(42)
random.shuffle(all_examples)
TRAIN_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 70
train_examples = all_examples[:TRAIN_SIZE]
test_examples  = all_examples[TRAIN_SIZE:]


def add_base3(a1, a0, b1, b0):
    a = a1 * BASE + a0
    b = b1 * BASE + b0
    s = a + b
    d2 = s // (BASE * BASE)
    d1 = (s // BASE) % BASE
    d0 = s % BASE
    return d2, d1, d0


def make_dataset(examples):
    inputs, targets = [], []
    for (a1, a0, b1, b0) in examples:
        d2, d1, d0 = add_base3(a1, a0, b1, b0)
        seq = [a1, a0, stoi['+'], b1, b0, stoi['='], d2, d1, d0]
        inp = seq[:-1]
        tgt = seq[1:]
        masked_tgt = [-100] * len(tgt)
        # In the shifted target, output digits d2,d1,d0 sit at indices 5,6,7
        masked_tgt[5] = tgt[5]
        masked_tgt[6] = tgt[6]
        masked_tgt[7] = tgt[7]
        inputs.append(inp)
        targets.append(masked_tgt)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def evaluate(model, examples):
    model.eval()
    correct, total = 0, 0
    rows = []
    with torch.no_grad():
        for (a1, a0, b1, b0) in examples:
            d2, d1, d0 = add_base3(a1, a0, b1, b0)
            expected = [d2, d1, d0]
            ctx = [a1, a0, stoi['+'], b1, b0, stoi['=']]
            idx = torch.tensor([ctx], device=device)
            preds = []
            for _ in range(3):
                logits, _ = model(idx)
                next_tok = logits[0, -1, :].argmax().item()
                idx = torch.cat([idx, torch.tensor([[next_tok]], device=device)], dim=1)
                preds.append(next_tok)
            ok = preds == expected
            correct += int(ok)
            total += 1
            rows.append((a1, a0, b1, b0, preds, expected, ok))
    model.train()
    return correct, total, rows


# ---------- BASE-parameterized constants (from TinyAdder / AdderBoard) ----------
# For BASE=3: DIGIT_SCALE=3, FINAL_SCALE=9, PLACE_VALUE=3
DIGIT_SCALE = BASE
FINAL_SCALE = BASE ** 2
# Operand value extraction from one-hot: val = x[dim+1] + 2*x[dim+2]
# s = 3*(a1+b1) + (a0+b0) for sum; d2=s//9, d1=(s//3)%3, d0=s%3


def _get_ffn_input(m, ctx, device):
    """Run forward up to (including) ln2, return the vector at last position for FFN input."""
    idx = torch.tensor([ctx], device=device)
    tok = m.token_embedding_table(idx)
    pos = m.position_embedding_table(torch.arange(len(ctx), device=device))
    x = tok + pos
    # Block 0: attention only (FFN zeroed)
    x = x + m.blocks[0].sa(m.blocks[0].ln1(x))
    x_ln = m.blocks[0].ln2(x)
    return x_ln[0, -1].clone()


def build_hand_ternary_2digit():
    """Hand-designed 2-digit ternary adder using AdderBoard / TinyAdder ideas.

    Design (from research):
    1. 4-head attention: each head copies one operand (a1, a0, b1, b0) into residual.
    2. FFN: computes digit from (a1,a0,b1,b0,step) via calibrated ReLU gates
       (TinyAdder "V-shape" / lookup style; formula: s=3*(a1+b1)+(a0+b0),
        d2=s//9, d1=(s//3)%3, d0=s%3).
    3. lm_head: reads digit logits from FFN output dims 0-2.

    Layout (n_embd=24):
      0-4:   token one-hots
      5-8:   operand markers (a1,a0,b1,b0)
      9-11:  step markers (d2,d1,d0)
      12-23: operand digit one-hots from attention
    """
    head_size = n_embd // n_head
    m = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)

    with torch.no_grad():
        # Token embedding: one-hot for 0,1,2,+,=
        tok_emb = torch.zeros(vocab_size, n_embd)
        for i in range(5):
            tok_emb[i, i] = 1.0
        m.token_embedding_table.weight.copy_(tok_emb)

        # Position embedding: operand + step markers
        pos_emb = torch.zeros(block_size, n_embd)
        pos_emb[0, 5] = 1.0
        pos_emb[1, 6] = 1.0
        pos_emb[3, 7] = 1.0
        pos_emb[4, 8] = 1.0
        pos_emb[5, 9] = 1.0
        pos_emb[6, 10] = 1.0
        pos_emb[7, 11] = 1.0
        m.position_embedding_table.weight.copy_(pos_emb)

        # 4 heads: copy a1, a0, b1, b0 into residual
        operand_markers = [5, 6, 7, 8]
        for h_idx in range(n_head):
            head = m.blocks[0].sa.heads[h_idx]
            W_Q = torch.zeros(head_size, n_embd)
            for s in [9, 10, 11]:
                W_Q[0, s] = 10.0
            head.query.weight.copy_(W_Q)
            W_K = torch.zeros(head_size, n_embd)
            W_K[0, operand_markers[h_idx]] = 10.0
            head.key.weight.copy_(W_K)
            W_V = torch.zeros(head_size, n_embd)
            for d in range(BASE):
                W_V[d, d] = 1.0
            head.value.weight.copy_(W_V)

        W_proj = torch.zeros(n_embd, n_embd)
        for h_idx in range(n_head):
            src_offset = h_idx * head_size
            dst_offset = 12 + h_idx * BASE
            for d in range(BASE):
                W_proj[dst_offset + d, src_offset + d] = 1.0
        m.blocks[0].sa.proj.weight.copy_(W_proj)
        m.blocks[0].sa.proj.bias.zero_()

        # LayerNorms: identity-like
        m.blocks[0].ln1.weight.fill_(1.0)
        m.blocks[0].ln1.bias.zero_()
        m.blocks[0].ln2.weight.fill_(1.0)
        m.blocks[0].ln2.bias.zero_()
        m.ln_f.weight.fill_(1.0)
        m.ln_f.bias.zero_()

    # Collect FFN inputs (ln2 output) for all 243 (input, step) pairs
    m.eval()
    ffn_inputs, targets = [], []
    with torch.no_grad():
        for a1 in range(BASE):
            for a0 in range(BASE):
                for b1 in range(BASE):
                    for b0 in range(BASE):
                        d2, d1, d0 = add_base3(a1, a0, b1, b0)
                        ctx = [a1, a0, stoi['+'], b1, b0, stoi['=']]
                        ffn_inputs.append(_get_ffn_input(m, ctx, device))
                        targets.append(d2)
                        ffn_inputs.append(_get_ffn_input(m, ctx + [d2], device))
                        targets.append(d1)
                        ffn_inputs.append(_get_ffn_input(m, ctx + [d2, d1], device))
                        targets.append(d0)

    X_ffn = torch.stack(ffn_inputs)
    target_tensor = torch.tensor(targets, device=device, dtype=torch.long)

    # Calibrate FFN: learn W1, b1, W2, b2 to map X_ffn -> digit logits in dims 0-2
    # FFN: h = ReLU(X @ W1 + b1), out = h @ W2 + b2; we want out[:, 0:3] to classify correctly
    ffn = m.blocks[0].ffwd
    ffn.net[0].weight.data.zero_()
    ffn.net[0].bias.data.zero_()
    ffn.net[2].weight.data.zero_()
    ffn.net[2].bias.data.zero_()

    # Only train output dims 0-2 (digit logits); keep W2[3:,:] zero so FFN doesn't pollute other dims
    W1, b1 = ffn.net[0].weight, ffn.net[0].bias
    W2, b2 = ffn.net[2].weight, ffn.net[2].bias
    params = [W1, b1, W2, b2]
    for p in params:
        p.requires_grad_(True)

    optimizer_ffn = torch.optim.Adam([W1, b1, W2, b2], lr=0.01)
    for step in range(10000):
        optimizer_ffn.zero_grad()
        h = F.relu(X_ffn @ W1.T + b1)
        out = h @ W2.T + b2
        out_digits = out[:, 0:3]
        loss = F.cross_entropy(out_digits, target_tensor)
        loss.backward()
        # Only update W2 rows 0-2 and b2[:3]; zero grads for the rest
        with torch.no_grad():
            if W2.grad is not None:
                W2.grad[3:, :] = 0.0
            if b2.grad is not None:
                b2.grad[3:] = 0.0
        optimizer_ffn.step()
        with torch.no_grad():
            W2.data[3:, :] = 0.0
            b2.data[3:] = 0.0
        if loss.item() < 1e-6:
            break

    with torch.no_grad():
        h = F.relu(X_ffn @ W1.T + b1)
        out = h @ W2.T + b2
        out_digits = out[:, 0:3]
    ffn_acc = (out_digits.argmax(dim=1) == target_tensor).float().mean().item()
    print(f"  [hand] FFN calibration: {step} steps, acc={ffn_acc:.4f}")

    # Scale FFN output so it dominates residual (embedding is ~1, we want digit signal >> 1)
    with torch.no_grad():
        ffn.net[2].weight.data[0:3, :] *= 10.0
        ffn.net[2].bias.data[0:3] *= 10.0

    # lm_head: read dims 0-2 for digit logits
    with torch.no_grad():
        m.lm_head.weight.zero_()
        m.lm_head.bias.zero_()
        for d in range(3):
            m.lm_head.weight[d, d] = 1.0

    # Verify end-to-end: run full forward and check lm_head
    m.eval()
    vectors, targets2 = [], []
    with torch.no_grad():
        for a1 in range(BASE):
            for a0 in range(BASE):
                for b1 in range(BASE):
                    for b0 in range(BASE):
                        d2, d1, d0 = add_base3(a1, a0, b1, b0)
                        for ctx, tgt in [
                            ([a1, a0, stoi['+'], b1, b0, stoi['=']], d2),
                            ([a1, a0, stoi['+'], b1, b0, stoi['='], d2], d1),
                            ([a1, a0, stoi['+'], b1, b0, stoi['='], d2, d1], d0),
                        ]:
                            idx = torch.tensor([ctx], device=device)
                            tok = m.token_embedding_table(idx)
                            pos = m.position_embedding_table(torch.arange(len(ctx), device=device))
                            x = tok + pos
                            x = m.blocks(x)
                            x = m.ln_f(x)
                            vectors.append(x[0, -1].clone())
                            targets2.append(tgt)

    X_final = torch.stack(vectors)
    target_final = torch.tensor(targets2, device=device)
    logits = X_final @ m.lm_head.weight.T + m.lm_head.bias
    acc_final = (logits.argmax(dim=1) == target_final).float().mean().item()
    print(f"  [hand] End-to-end lm_head acc (dims 0-2): {acc_final:.4f}")

    return m


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 60)
    print("  GENERALIZATION TEST: 2-DIGIT TERNARY ADDITION")
    print("=" * 60)
    print(f"\n  Train set: {len(train_examples)} examples")
    print(f"  Test set:  {len(test_examples)} examples (NEVER seen during training)")

    hand_model = build_hand_ternary_2digit()

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
    print("  4-WAY COMPARISON: 2-DIGIT TERNARY ADDITION")
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
        correct = sum(1 for r in results if r[6])
        total = len(results)
        print(f"\n  {label} ({correct}/{total}):")
        for (a1, a0, b1, b0, preds, expected, ok) in results:
            pred_str = ''.join(itos[p] for p in preds)
            exp_str  = ''.join(itos[e] for e in expected)
            mark = "PASS" if ok else "FAIL"
            print(f"    {a1}{a0}+{b1}{b0} -> {pred_str} (expected {exp_str}) {mark}")
