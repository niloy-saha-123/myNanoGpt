# Final Report: Hand-Designed GPTs vs Trained GPTs on Algorithmic Tasks

COMP560 final project

## Summary

This project began as an attempt to understand GPTs deeply enough to build them by hand. It ended as a comparison between two very different ways of getting a tiny transformer to solve a task:

- directly hand-design the weights
- train the weights with SGD-style optimization

The central question was:

**If a tiny GPT can represent a computation exactly, when does training actually learn that computation instead of just memorizing examples?**

The final answer is that the architecture is often capable much earlier than the training process is. The hand-designed models show what the transformer can do. The trained experiments show when gradient-based learning succeeds, when it fails, and how much that depends on the amount of data.

## How the Project Developed

The hand-designed part of the project started with two single-pass transformer programs:

- `projects/hand_copy.py`
- `projects/hand_reverse.py`

Those models solved the tasks in one shot, but professor feedback pushed the project in a more realistic direction: use the transformer autoregressively, the same way GPTs actually generate text, one token at a time.

That led to a second and more important stage of the project:

- `projects/hand_copy_ar.py`
- `projects/hand_reverse_ar.py`
- `projects/hand_addition_ar.py`
- `projects/hand_ternary_addition_ar.py`
- `projects/hand_decimal_addition_ar.py`
- `projects/hand_markov_ar.py`

This shift mattered because it made the hand-designed models and the trained models part of the same story. Both were now studying the same kind of autoregressive transformer computation.

## Research Question

Can a tiny autoregressive GPT trained on a subset of examples generalize the actual rule on simple algorithmic tasks, and how does that compare to a tiny GPT whose weights are designed directly to perform the computation?

## Method

I used the same general decoder-only transformer idea in two ways.

### 1. Hand-designed GPTs

In `projects/`, I manually chose the token embeddings, position embeddings, attention weights, feed-forward weights, and output logic for tiny 1-layer, 1-head GPTs. These models were built analytically rather than trained.

They were tested on:

- copy
- reverse
- binary addition
- ternary addition
- decimal addition
- a simple 3-state Markov chain

### 2. Trained GPTs

In `research/`, I trained small GPTs on subsets of the same kinds of tasks and evaluated them on unseen examples. These experiments measured whether the model learned the rule itself or only memorized the training set.

The main trained tasks were:

- copy
- reverse
- 1-digit addition
- 2-digit addition
- 3-digit addition
- Markov sequence prediction

## What the Hand-Designed GPTs Showed

The hand-designed models showed that the architecture itself is expressive enough to solve these tasks.

The most important technical lesson from that part of the project came from arithmetic. A first attempt at binary addition failed because it encoded the sum as a scalar magnitude in one hidden dimension. LayerNorm destroyed that magnitude information. After normalization, different sums looked almost identical.

The fix was to encode information as a pattern across dimensions rather than a single number in one coordinate. That made the arithmetic models work and turned LayerNorm from an annoyance into a central part of the research finding.

The hand-designed results were:

- single-pass copy: exact
- single-pass reverse: exact
- autoregressive copy: exact
- autoregressive reverse: exact
- binary addition: exact on all cases
- ternary addition: exact on all cases
- decimal addition: `100/100`
- Markov chain: correct qualitative transition behavior

The Markov result was also useful because it showed that the same tiny transformer architecture could be hand-programmed not just for deterministic string transformations but also for probabilistic next-token behavior.

This side of the project established an important baseline: if the trained model fails, that failure is not automatically because the architecture is too weak. In many cases, the model class can do the task exactly.

## Baseline Trained Results

The first trained comparison used small datasets:

- copy on `3^3 = 27` strings, with `9` train and `18` held out
- reverse on the same space, also `9` train and `18` held out
- 1-digit decimal addition, with `50` train and `50` held out

These models fit the seen examples well, but generalization was much weaker:

| Task | Training set | Held-out result |
|---|---:|---:|
| Copy | 9 | `16/18 = 88.9%` |
| Reverse | 9 | `11/18 = 61.1%` |
| 1-digit addition | 50 | about `16-22%` |

At this stage, the rough pattern was:

- copy was easiest
- reverse was harder
- addition was much harder

That suggested that routing-style tasks are easier for SGD to learn from little data than arithmetic-style tasks that require more structured internal computation.

## Why the Study Was Extended

Stopping at those baseline numbers would have made the conclusion too simplistic. The hand-designed models had already shown that reverse and addition were possible. So the more interesting question became whether the trained failures were really about task impossibility or just about data scarcity.

That led to the extension experiments:

- longer copy tasks
- longer reverse tasks
- larger 1-digit addition datasets
- 2-digit and 3-digit addition
- variable-data Markov learning

The goal of the extensions was to test whether the same small GPTs would improve once they were given enough evidence to learn the rule rather than memorize examples.

## Extension Results

The strongest extension results were:

| Task | Configuration | Held-out result |
|---|---|---:|
| Copy | 4-letter, 32 train | `100%` |
| Copy | 5-letter, 500 train | `100%` |
| Copy | 8-letter, 1000 train | `100%` |
| Reverse | 5-letter, 500 train | `100%` |
| Reverse | 8-letter, 2000 train | `100%` |
| 1-digit addition | 75 train | `68%` |
| 2-digit addition | 2000 train | `88.7%` |
| 3-digit addition | 1000 train | `820/2000 = 41.0%` |
| Markov | 256 or 5000 sequences | `KL ≈ 0.002` |

These results changed the interpretation of the project in an important way.

Reverse did not fail because transformers fundamentally cannot reverse. It failed in the original small-data regime, but with enough examples it reached `100%`.

Addition behaved similarly. With `50` training examples, it generalized very poorly. With more data, performance improved a lot. By the 2-digit setting, the same basic model family was reaching `88.7%` on unseen problems. Even 3-digit addition, which is much harder, showed clear partial rule learning at `41.0%`.

The Markov experiment also supported the same theme. The trained model was able to approximate the transition structure increasingly well as the amount of training data increased.

## Main Findings

The final findings of the project are:

1. **Tiny GPTs can represent these computations.**  
   The hand-designed models prove that the architecture can exactly implement copy, reverse, arithmetic, and simple Markov behavior.

2. **Learning and representation are different problems.**  
   A model class can be fully capable of a task even when SGD does not reliably discover the computation from small data.

3. **LayerNorm matters conceptually, not just numerically.**  
   The arithmetic hand-designs showed that transformer internals place real constraints on what kinds of signals survive through the network. That made the hand-designed part a substantive research result rather than a coding trick.

4. **Data size changes the conclusion.**  
   The baseline experiments alone made reverse and addition look much less learnable than they really were. The extension runs showed that more data substantially changes the trained model's behavior.

5. **Copy is easiest, reverse is intermediate, and addition is hardest under limited data.**  
   But the harder tasks are not outside the model's reach. They simply require more evidence before training begins to recover the actual rule.

## Overall Conclusion

The most important outcome of the project is the gap it makes visible:

**there is a real difference between what a transformer can represent and what training will actually learn from limited data.**

The hand-designed GPTs answered the capacity question. The trained GPTs answered the learning question. Taken together, they produced a clearer and more defensible conclusion than either part would have alone.

That is the final contribution of the project: not just that tiny transformers can solve simple algorithmic tasks, but that hand-designing and training them side by side reveals when failures come from the architecture and when they come from the learning process.

## Reproducibility

Environment:

```bash
conda activate comp560
cd myNanoGpt
```

Run the main hand-designed autoregressive models:

```bash
cd projects
python hand_copy_ar.py
python hand_reverse_ar.py
python hand_addition_ar.py
python hand_ternary_addition_ar.py
python hand_decimal_addition_ar.py
python hand_markov_ar.py
```

Run the main trained experiments:

```bash
cd research
python generalize_copy.py
python generalize_reverse.py
python generalize_addition.py
python generalize_addition_moredata.py 75
python generalize_addition_2digit.py 2000
python generalize_addition_3digit.py 1000
python generalize_markov_variable.py 256
```

For the detailed hand-designed discussion, see:

- `projects/README.md`
- `projects/EXPLANATION.md`
