# Final Report: Hand-Designed GPTs vs Trained GPTs

COMP560 final project

## Overview

This project studied a simple question:

**What is the difference between a transformer that can represent a computation and a transformer that actually learns that computation from data?**

To answer that, I used the same small GPT-style architecture in two ways. In `projects/`, I built tiny transformers by hand by directly choosing the weights. In `research/`, I trained small transformers on subsets of the same kinds of tasks and tested whether they generalized to unseen examples.

## Hand-Designed GPTs

The hand-designed part of the project began with single-pass copy and reverse models, but after professor feedback I shifted to the autoregressive setting so the models would generate one token at a time, like a real GPT.

That led to hand-designed autoregressive models for:

- copy
- reverse
- binary addition
- ternary addition
- decimal addition
- a simple Markov chain

These models showed that a very small GPT can be programmed directly to carry out each task. The most important technical lesson was that LayerNorm matters a lot: for arithmetic, a naive design based on single-dimension magnitudes failed, and the model only worked once the information was encoded as a pattern across dimensions. This made the hand-designed side of the project more than a demonstration. It showed exactly how the transformer components constrain what kind of computation is possible.

## Trained GPTs

The trained side asked a different question. If the architecture can do the task, will SGD actually learn the rule from limited data?

The first experiments suggested a clear gap:

- copy generalized fairly well
- reverse generalized much worse
- addition generalized very poorly

This made it look as though the more structured the task was, the harder it was for a trained GPT to learn from small datasets.

## Extension Results

I then scaled the amount of training data and repeated the experiments on larger versions of the tasks.

The main results were:

- copy reached `100%` on larger versions
- reverse also reached `100%` once enough data was provided
- 1-digit addition improved substantially with more training examples
- 2-digit addition reached `88.7%`
- 3-digit addition reached `41.0%`
- the Markov experiment learned the transition structure well, with `KL ≈ 0.002`

These extension runs changed the conclusion. Reverse and addition were not failing because transformers fundamentally could not do them. They were failing because learning the rule from a small amount of data was difficult.

## Main Finding

The main finding of the project is:

**there is a real gap between what a tiny transformer can represent and what training will actually learn from limited data.**

The hand-designed GPTs proved that the architecture could solve these tasks. The trained GPTs showed that learning those same computations depends strongly on how much data the model sees.

That means the interesting question is not just whether transformers can do a task, but when training discovers the underlying computation instead of memorizing examples.

## Conclusion

Taken together, the two parts of the project produced a stronger result than either one alone. The hand-designed models established capacity. The trained models measured learning. Comparing them side by side made it possible to separate architectural ability from optimization and data effects.

The final conclusion is that tiny GPTs can represent simple algorithmic tasks very well, but training only learns those rules reliably once the data regime is large enough. That gap between representational power and learned generalization is the central result of the project.
