# Hand-Designed GPT Projects (COMP560)

This is my project for COMP560 where I'm figuring out how to design GPT models by hand — no training, no gradient descent. Just setting the weights myself and seeing if the model does what I want.

The idea: if we really understand how transformers work, we should be able to write down the right numbers ourselves for simple tasks. Turns out it's doable but kinda tricky in places.

---

## How we got here

I started with two single-pass models (`hand_copy.py` and `hand_reverse.py`) that take the full input and spit out the full output in one shot. My professor reviewed them and said something like: *"I imagined applying the transformer according to the methodology of generating one token at a time, sequentially. Both are interesting, but the sequential approach has more application to tasks like addition."*

So I switched to the autoregressive approach — generate one token at a time, like a real GPT. All the `_ar` files do that.

---

## The files

### Single-pass (original)

- **hand_copy.py** — Input `ABC`, output `ABC`. 27/27 correct. Identity embeddings, zero attention/FFN, residual carries everything through.
- **hand_reverse.py** — Input `ABC`, output `CBA`. 27/27 correct. Anti-diagonal attention so position 0 reads from position 2, etc. No causal mask because we need to look "forward" in the input.

### Autoregressive (token-by-token)

- **hand_copy_ar.py** — Input `ABC <sep>`, output `ABC` one token at a time. Works by having each output position attend back 3 positions to the right input slot.
- **hand_reverse_ar.py** — Input `ABC <sep>`, output `CBA` one token at a time. Same anti-diagonal idea but in autoregressive form.
- **hand_markov_ar.py** — Simulates a Markov chain over A,B,C. The model predicts the next token based on the previous one with fixed transition probabilities. Probabilities get a bit distorted by LayerNorm but the behavior is right.
- **hand_addition_ar.py** — Binary addition: `a + b =` → one output digit (0 or 1). 4/4 correct. This one was rough to get working (see below).
- **hand_ternary_addition_ar.py** — Ternary addition: `a + b =` → two output digits in base 3. 18/18 correct.
- **hand_decimal_addition_ar.py** — Decimal addition: `a + b =` → two output digits. 100/100 correct. Had to fix a calibration bug (see below).

---

## Results summary

| Task | Result |
|------|--------|
| Copy (single-pass) | 27/27 |
| Reverse (single-pass) | 27/27 |
| Copy (AR) | works |
| Reverse (AR) | works |
| Markov (AR) | works (probs slightly off) |
| Binary addition (AR) | 4/4 |
| Ternary addition (AR) | 18/18 |
| Decimal addition (AR) | 100/100 |

---

## Stuff that broke and how we fixed it

### Binary addition — 1/4 correct at first

For a while only `0+0=0` worked. The other three (`0+1`, `1+0`, `1+1`) failed. A few things were wrong:

1. **Dimension mismatch** — FFN had `hidden=20` but the architecture expected `4*n_embd=32`. Fixed by using the right size.
2. **Transposed indices** — PyTorch's `nn.Linear` stores weights as `[out, in]`. I had the indices backwards for W_Q and W_V, so we were reading/writing the wrong dimensions.
3. **LayerNorm killed the signal** — The biggest one. I was storing the sum (0, 50, or 100) in a single dimension. LayerNorm normalizes to zero mean and unit variance, so 50 and 100 both ended up looking the same. The fix: instead of one dimension with magnitudes, use two dimensions (one for "saw 0", one for "saw 1") and let the FFN classify based on the *pattern* of those two. Patterns survive LayerNorm; raw magnitudes don't.

### Decimal addition — 98/100 at first

Two cases failed: `3+9=12` and `9+3=12`. The lm_head calibration (iterative optimization to set the final classification weights) hadn't converged enough. Bumped iterations from 5000 to 20000 and got 100/100.

---

## What I learned

- **Attention = routing** — Q and K decide *where* to look, V carries *what* to pass through.
- **LayerNorm is sneaky** — It destroys absolute magnitudes. If you want to hand-design weights, encode information in *relative* patterns across dimensions, not in single-dimension magnitudes.
- **Autoregressive is the right framing** — Matches how real GPTs work and generalizes to tasks like addition where you output multiple digits.
- **Calibration works** — For ternary/decimal addition, we couldn't solve the lm_head weights by hand, so we used a small iterative optimization loop. Still no "training" in the usual sense — we're just solving for weights that make the fixed model output correct on a known dataset.

---

## Running the code

```bash
conda activate comp560
python hand_copy_ar.py
python hand_reverse_ar.py
python hand_addition_ar.py
python hand_ternary_addition_ar.py
python hand_decimal_addition_ar.py
python hand_markov_ar.py
```

Each one prints token-by-token output so you can see the model generating step by step.

---

For a deeper dive (theory, architecture, all the keywords explained), see `EXPLANATION.md`.
