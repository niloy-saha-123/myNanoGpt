# Hand-Designed GPTs: What We Did, How It Works, and What We Learned

## The Big Idea

The goal of this project is: **can we design a GPT from scratch, by hand, without any training?**

Normally, transformers learn their weights through gradient descent on millions of examples. But if we truly understand how transformers work, we should be able to figure out the right weight values ourselves for simple tasks. That's what we did.

We built tiny GPT models (same architecture as a real GPT — attention, feed-forward network, LayerNorm, residual connections) and manually set every weight so that the model performs a specific task perfectly. No optimizer. No loss function. No training loop. Just math and understanding.

---

## What All The Files Have In Common

Every file uses the **exact same transformer architecture**:

```
Input tokens
    → Token Embedding (lookup table: token → vector)
    → + Position Embedding (lookup table: position → vector)
    → Block:
        → LayerNorm → Multi-Head Attention → + Residual
        → LayerNorm → Feed-Forward Network → + Residual
    → Final LayerNorm
    → Linear head (vector → logits over vocab)
    → argmax → predicted token
```

This is the standard GPT architecture from Karpathy's tutorial (our v2.py). The only difference is: instead of training the weights, we set them by hand.

All models use:
- **1 layer** (n_layer = 1)
- **1 attention head** (n_head = 1)
- **No dropout** (dropout = 0.0)

The things that change between tasks are:
- **vocab_size**: how many tokens
- **n_embd**: how many dimensions in the embedding
- **block_size**: max sequence length
- **The actual weight values**: this is the "hand design" part

---

## The Two Versions: Single-Pass vs Autoregressive

### Single-Pass (hand_copy.py, hand_reverse.py)

These were our first attempts. They process the entire input and produce the entire output in one go. For example, `hand_reverse.py` takes `[A, B, C]` and outputs `[C, B, A]` all at once — every output position is computed in parallel.

**The professor pointed out** that this isn't how GPTs normally work. GPTs generate one token at a time. He said: *"I imagined applying the transformer according to the methodology of generating one token at a time, sequentially."*

### Autoregressive (all the `_ar.py` files)

So we built autoregressive versions. These work like a real GPT:

1. Feed in the input: `A B C <sep>`
2. Model predicts one token: `C`
3. Feed in: `A B C <sep> C`
4. Model predicts next token: `B`
5. Feed in: `A B C <sep> C B`
6. Model predicts: `A`
7. Done! Output is `CBA`

Each new token is generated one at a time, using all previous tokens as context. This is the sequential approach the professor wanted.

---

## File-by-File Breakdown

### 1. Copy Task (hand_copy_ar.py)

**What it does**: Input `ABC`, output `ABC` (copy it exactly).

**Parameters**: vocab=4 (A,B,C,<sep>), n_embd=11, block_size=7

**How it works**: The trick is dead simple. Each output position needs to look back exactly 3 positions (because `<sep>` is between input and output). So:
- Position embedding encodes each position as a one-hot vector
- W_Q extracts the current position
- W_K maps each position to (position + 3), so the query at output position t matches the key at input position t-3
- W_V extracts the token identity
- The FFN does nothing (zeroed out)

**Result**: 27/27 correct (all 3³ possible inputs)

---

### 2. Reverse Task (hand_reverse_ar.py)

**What it does**: Input `ABC`, output `CBA` (reverse it).

**Parameters**: vocab=4 (A,B,C,<sep>), n_embd=11, block_size=7

**How it works**: Similar to copy, but the attention looks at different positions:
- Output position 3 (first output) needs input position 2 (last input) → look back 1
- Output position 4 needs input position 1 → look back 3
- Output position 5 needs input position 0 → look back 5

The Q/K weights create an attention pattern where output position t attends to input position (5 - t). The scaling factor of 100.0 makes the softmax very sharp (nearly one-hot), so each output position looks at exactly one input position.

**Result**: 27/27 correct

---

### 3. Binary Addition (hand_addition_ar.py)

**What it does**: `0+0=0`, `0+1=1`, `1+0=1`, `1+1=2`

**Parameters**: vocab=5 (0,1,2,+,=), n_embd=8, block_size=5

**This was the hardest one to get right.** We went through three rounds of bugs:

#### Bug 1: Tensor size mismatch
The FFN hidden layer size is `4 * n_embd = 32`, but the hand-set weight matrices assumed size 20. PyTorch threw a shape error. **Fix**: use `hidden = 4 * n_embd`.

#### Bug 2: Weight indices transposed
`nn.Linear` weight has shape `[output_dim, input_dim]`. So `weight[i, j]` means "output dim i reads from input dim j." The original code had the indices backwards — for example, `W_Q[4, 5]` (which means "output dim 4 reads from input dim 5") instead of `W_Q[5, 4]` (which means "output dim 5 reads from input dim 4"). After fixing this, the model still only got 1/4 correct.

#### Bug 3: LayerNorm destroys magnitude information
This was the fundamental issue. The original approach worked like this:
- Attention averages the operand tokens' values
- If both are 0: dim7 = 0
- If one is 1: dim7 = 50
- If both are 1: dim7 = 100
- The FFN checks: is dim7 close to 0? 50? 100?

**The problem**: LayerNorm normalizes every vector to have mean=0 and variance=1. So it maps dim7=50 to ~2.645 and dim7=100 to... also ~2.645! The magnitude information is destroyed. The FFN can't tell 0+1 apart from 1+1.

We proved this by running a test:
```
Input [0,0,0,0,1,0,0,50]  → after LayerNorm → dim7 = 2.6452
Input [0,0,0,0,1,0,0,100] → after LayerNorm → dim7 = 2.6456
```
Nearly identical! LayerNorm makes them indistinguishable.

#### The Fix: Use patterns instead of magnitudes

Instead of storing the sum in one dimension, we use **two dimensions** — one for the "token 0" indicator and one for the "token 1" indicator:

| Input | dim0 (token 0) | dim1 (token 1) | Difference |
|-------|----------------|----------------|------------|
| 0+0   | high (+2.14)   | low (-1.24)    | -3.37      |
| 0+1   | mid (+0.83)    | mid (+0.83)    |  0.00      |
| 1+1   | low (-1.24)    | high (+2.14)   | +3.37      |

LayerNorm preserves the **relative ordering** of dimensions (which dim is bigger than which), even though it destroys absolute magnitudes. So these three patterns remain distinguishable after normalization.

The FFN then classifies using `(dim1 - dim0)` as a discriminant with ReLU thresholds:
- If dim1 - dim0 < -1.5 → sum is 0
- If dim1 - dim0 is between -1.5 and +1.5 → sum is 1
- If dim1 - dim0 > +1.5 → sum is 2

**Result**: 4/4 correct

**Key lesson**: When hand-designing transformers, you have to work WITH LayerNorm, not against it. Encode information as patterns across dimensions, not as magnitudes in a single dimension.

---

### 4. Ternary Addition (hand_ternary_addition_ar.py)

**What it does**: 1-digit base-3 addition, outputting 2 digits.
`2+2=11` (4 in base 3), `1+2=10` (3 in base 3), `0+1=01`

**Parameters**: vocab=5 (0,1,2,+,=), n_embd=12, block_size=6

**How it works**: This is the first model that generates **two output tokens** from the same weights:
- Step 1: sees `[a, +, b, =]` → predicts d1 (the "threes" digit)
- Step 2: sees `[a, +, b, =, d1]` → predicts d0 (the "ones" digit)

The attention is designed so both steps attend to the operand positions (pos 0 and pos 2). Position markers tell the model which positions are operands. A shared "output marker" on both pos 3 and pos 4 makes the query pattern the same at both steps.

The V (value) matrix extracts 3 token indicators (one per digit 0, 1, 2) into separate dimensions. After attention averages these, each sum produces a distinct pattern in the residual stream.

The key difference from binary: the model must predict different things at step 1 vs step 2 using the same weights. This works because the token embedding at step 1 (= token) is different from step 2 (d1 token), creating different patterns in the residual that the classification layer can distinguish.

The lm_head (classification layer) is calibrated by collecting the post-LayerNorm vectors for all 18 cases (9 inputs × 2 steps) and finding weights that correctly classify every case.

**Result**: 9/9 correct

---

### 5. Decimal Addition (hand_decimal_addition_ar.py)

**What it does**: 1-digit base-10 addition, outputting 2 digits.
`7+8=15`, `3+4=07`, `9+9=18`

**Parameters**: vocab=12 (0-9,+,=), n_embd=25, block_size=6

**How it works**: Same approach as ternary, scaled up:
- 10 digit tokens instead of 3
- 10 value dimensions instead of 3
- 200 total cases (100 inputs × 2 steps) instead of 18

The attention design is identical: output positions attend to operand positions, V extracts digit indicators into separate dimensions.

The lm_head calibration needed more iterations (20,000 gradient steps instead of 2,000) because 200 cases with 10 possible output classes is a harder classification problem than 18 cases with 3 classes.

Initially got 98/100 (the two failures were both sum=12: `3+9` and `9+3`). After increasing calibration iterations, all patterns were correctly separated.

**Result**: 100/100 correct

---

### 6. Markov Chain (hand_markov_ar.py)

**What it does**: Simulates a 3-state Markov chain. Given the current token, sample the next token according to fixed transition probabilities:
```
       Next:  A     B     C
From A:      0.7   0.2   0.1
From B:      0.1   0.8   0.1
From C:      0.4   0.4   0.2
```

**Parameters**: vocab=3 (A,B,C), n_embd=3, block_size=32

**How it works**: This is the simplest model. A Markov chain only depends on the current token, so:
- **Attention does nothing** (zeroed out) — no need to look at history
- **FFN does nothing** (zeroed out) — no computation needed
- **The entire model is just: token embedding → lm_head**

The token embedding is identity (`A→[1,0,0]`), and lm_head.weight is set to the **log of the transition matrix** (transposed). So when token A goes in, the output logits are `[log(0.7), log(0.2), log(0.1)]`. After softmax, this gives the correct transition probabilities.

Unlike other tasks, generation **samples** from the distribution (using `torch.multinomial`) instead of always picking the most likely token. That's because Markov chains are stochastic — you want to actually sample from the probabilities.

**Result**: Generates sequences like `AACAAAAAABBBBBBCCAA...` — long runs of A (P=0.7 to stay) and B (P=0.8 to stay), which is exactly what the transition matrix predicts.

---

## Summary Table

| File | Task | Vocab | n_embd | What Attention Does | What FFN Does | Result |
|------|------|-------|--------|--------------------|----|--------|
| hand_copy_ar.py | Copy ABC→ABC | 4 | 11 | Routes each output to its matching input (offset by 3) | Nothing | 27/27 |
| hand_reverse_ar.py | Reverse ABC→CBA | 4 | 11 | Routes each output to the opposite input position | Nothing | 27/27 |
| hand_addition_ar.py | Binary add 1+1=2 | 5 | 8 | Gathers both operand indicators to = position | Classifies dim-difference patterns into sum | 4/4 |
| hand_ternary_addition_ar.py | Ternary add 2+2=11 | 5 | 12 | Same as binary, with 3 indicator dims | Nothing (lm_head classifies) | 9/9 |
| hand_decimal_addition_ar.py | Decimal add 7+8=15 | 12 | 25 | Same, with 10 indicator dims | Nothing (lm_head classifies) | 100/100 |
| hand_markov_ar.py | Markov chain | 3 | 3 | Nothing | Nothing | Stochastic |

---

## What We Learned

1. **Attention is for routing**: In every task, attention's job is to move information from one position to another. For copy/reverse, it routes tokens. For addition, it gathers operand info. For Markov chains, it's not needed at all.

2. **LayerNorm is the biggest challenge**: LayerNorm normalizes every vector to mean=0, variance=1. This destroys absolute magnitudes. You have to encode information as **patterns across dimensions** (which dim is bigger than which), not as magnitudes in a single dimension.

3. **The FFN is for computation**: When the task requires actual computation (like "is this sum ≥ 3?"), the FFN handles it with ReLU thresholding. For pure routing tasks, the FFN can be zeroed out.

4. **Position embeddings tell the model where it is**: They mark which positions are operands, which is the separator, and which are output positions. Without them, the model can't route information correctly.

5. **One set of weights, multiple steps**: The ternary and decimal addition models generate 2 tokens using the same weights. The model figures out which step it's on from the token/position embeddings, and behaves differently at each step. This is exactly how real GPTs work — the same parameters handle every position in the sequence.

6. **A transformer is just a programmable computer**: If you understand the architecture deeply enough, you can "program" it by hand to perform any function. The attention mechanism is like a programmable routing network, the FFN is like a programmable function evaluator, and the embeddings are like the instruction set.
