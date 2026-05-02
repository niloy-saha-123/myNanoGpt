# myNanoGpt

This repo started from Andrej Karpathy's GPT-from-scratch code, but the final COMP560 project became a study of a more specific question:

**What is the difference between a transformer that can represent a computation and a transformer that actually learns that computation from data?**

I explored that in two connected parts:

- `projects/` contains hand-designed tiny GPTs with manually chosen weights
- `research/` contains trained GPT experiments on the same kinds of tasks

The main result of the repo is that tiny transformers are often capable of the computation long before SGD reliably learns it.

## Project Summary

The hand-designed side began with single-pass copy and reverse models, then shifted to the more realistic autoregressive setting after professor feedback. That led to tiny GPTs for:

- copy
- reverse
- binary addition
- ternary addition
- decimal addition
- a simple Markov chain

Those models show that the same basic GPT architecture can be programmed directly for these tasks.

The trained side uses the same general model family but learns from examples instead of hand-set weights. Those experiments test whether the model learns the actual rule or mainly memorizes the training set.

Together, the two parts show the main gap of the project:

**a transformer may be able to do a task, but training does not automatically learn that computation from limited data.**

## Main Findings

- hand-designed GPTs solved copy, reverse, arithmetic, and Markov-style next-token behavior
- the arithmetic hand-designs showed that LayerNorm is a real constraint, not just an implementation detail
- trained GPTs often memorized under small data
- with more data, the same small trained models began to generalize much better
- reverse eventually reached `100%`
- 2-digit addition reached `88.7%`
- 3-digit addition reached `41.0%`

## Repo Layout

### Core GPT code

- `bigram.py`
- `v2.py`
- `EXPLAIN.md`

### Hand-designed GPT project

- `projects/hand_copy.py`
- `projects/hand_reverse.py`
- `projects/hand_copy_ar.py`
- `projects/hand_reverse_ar.py`
- `projects/hand_addition_ar.py`
- `projects/hand_ternary_addition_ar.py`
- `projects/hand_decimal_addition_ar.py`
- `projects/hand_markov_ar.py`
- `projects/README.md`
- `projects/EXPLANATION.md`

### Trained generalization experiments

- `research/generalize_copy.py`
- `research/generalize_reverse.py`
- `research/generalize_addition.py`
- `research/generalize_addition_moredata.py`
- `research/generalize_addition_2digit.py`
- `research/generalize_addition_3digit.py`
- `research/generalize_markov_variable.py`
- `research/REPORT.md`

## Running the Main Experiments

```bash
conda activate comp560
cd myNanoGpt
```

Hand-designed autoregressive GPTs:

```bash
cd projects
python hand_copy_ar.py
python hand_reverse_ar.py
python hand_addition_ar.py
python hand_ternary_addition_ar.py
python hand_decimal_addition_ar.py
python hand_markov_ar.py
```

Main trained experiments:

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

## Final Report

The final project report is here:

- `research/REPORT.md`

The detailed hand-designed discussion is here:

- `projects/README.md`
- `projects/EXPLANATION.md`
