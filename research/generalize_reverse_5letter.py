"""
Generalization Test: Reverse Task (5 letters)

Same format as base: INPUT<sep> -> OUTPUT reversed.
5 letters (A..E), 5 positions. Input space: 5^5 = 3,125.

Shows progression: 3-letter (61%) -> 5-letter -> 8-letter (100%).

Usage:
  python generalize_reverse_5letter.py           # default 500 train
  python generalize_reverse_5letter.py 1000      # 1k train
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import os
import random
from model import GPT

NUM_LETTERS = 5
SEQ_LEN = NUM_LETTERS
vocab_size = NUM_LETTERS + 1
block_size = SEQ_LEN + 1 + SEQ_LEN  # 11
n_embd     = vocab_size + block_size
n_head     = 1
n_layer    = 1
device     = 'cpu' if os.environ.get('RESEARCH_USE_CPU') else ('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

itos = {i: chr(ord('A') + i) for i in range(NUM_LETTERS)}
itos[NUM_LETTERS] = '<sep>'
stoi = {v: k for k, v in itos.items()}

lr         = 1e-3
max_iters  = 12000
eval_every = 1200

TRAIN_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TEST_SIZE  = 1000
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
            if preds == expected:
                correct += 1
            total += 1
    model.train()
    return correct, total


def build_hand_reverse():
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
        for t in range(SEQ_LEN + 1, block_size):
            W_K[vocab_size + t, vocab_size + (block_size - 1 - t)] = 100.0
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
    all_examples = []
    for _ in range(TRAIN_SIZE + TEST_SIZE):
        all_examples.append(tuple(random.randint(0, NUM_LETTERS - 1) for _ in range(SEQ_LEN)))
    train_examples = all_examples[:TRAIN_SIZE]
    test_examples = all_examples[TRAIN_SIZE:]

    print("=" * 60)
    print("  GENERALIZATION TEST: Reverse Task (5 letters)")
    print("=" * 60)
    print(f"\n  Input space: {NUM_LETTERS}^{SEQ_LEN} = {NUM_LETTERS**SEQ_LEN:,}")
    print(f"  Train: {len(train_examples)}, Test: {len(test_examples)}")

    hand_model = build_hand_reverse()
    trained_model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    optimizer = torch.optim.AdamW(trained_model.parameters(), lr=lr)
    train_inputs, train_targets = make_dataset(train_examples)
    train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)

    print(f"\nTraining...\n")
    for step in range(max_iters + 1):
        logits, loss = trained_model(train_inputs, train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % eval_every == 0:
            tr_c, tr_t = evaluate(trained_model, train_examples)
            te_c, te_t = evaluate(trained_model, test_examples)
            print(f"  step {step:5d} | loss {loss.item():.4f} | train {tr_c}/{tr_t} | TEST {te_c}/{te_t}")

    h_seen_c, h_seen_t = evaluate(hand_model, train_examples)
    h_unseen_c, h_unseen_t = evaluate(hand_model, test_examples)
    t_seen_c, t_seen_t = evaluate(trained_model, train_examples)
    t_unseen_c, t_unseen_t = evaluate(trained_model, test_examples)
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Hand:    seen {h_seen_c}/{h_seen_t}  unseen {h_unseen_c}/{h_unseen_t}")
    print(f"  Trained: seen {t_seen_c}/{t_seen_t}  unseen {t_unseen_c}/{t_unseen_t}")
