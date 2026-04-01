# Hand-Designed vs. Trained Transformers: Generalization from Small to Large Data

COMP560 Research Report  
March 2025  

---

## 1. Overview

This project compares hand-designed and SGD-trained GPT models on simple algorithmic tasks. The architecture is the same in both cases: 1 layer, 1 attention head, no dropout. The difference is how the weights are set. Hand-designed models have weights derived analytically to encode the task; trained models learn from examples via AdamW. The main question is whether training on a subset of inputs leads to generalization to unseen inputs or only memorization.

---

## 2. Phase 1: Baseline Experiments

**Setup.** Three tasks: copy (repeat input), reverse (reverse order), and decimal addition (1-digit sums). For copy and reverse, the input space is 27 (3 letters in 3 positions); 9 examples for training, 18 held out for testing. For addition, 100 possible sums; 50 train, 50 test. Hand-designed models are built by setting attention keys and values so each output position attends to the correct input position, plus identity embeddings and output mapping.

**Results.** Hand-designed models get 100% on both seen and unseen inputs. Trained models all fit the training set, but generalize differently:

| Task    | Trained on | Unseen accuracy |
|---------|------------|-----------------|
| Copy    | 9          | 16/18 (88.9%)   |
| Reverse | 9          | 11/18 (61%)     |
| Addition| 50         | 8–11/50 (16–22%)|

Copy nearly generalizes; two failures. Reverse captures some patterns but fails on many unseen inputs. Addition shows little generalization and appears to be memorizing. The working hypothesis was that task complexity drives generalization: simple routing (copy) works, compositional reasoning (addition) does not.

---

## 3. Phase 2: Extension with More Data

The paper "Transformers Can Do Arithmetic with the Right Embeddings" (McLeish et al., NeurIPS 2024) argues that transformers can learn arithmetic under suitable setups. That led to scaling up training data and input spaces to test whether more examples improve generalization.

**Extended tasks.** Copy at 4, 5, and 8 letters; reverse at 5 and 8 letters; addition with 75 examples and 2-digit sums (2000 train); a 3-state Markov chain (predict next token from a fixed transition matrix); and ternary 2-digit addition. Training sizes were increased and, where relevant, made configurable via command line.

**Results.** Larger training sets changed generalization:

| Task   | Config              | Unseen accuracy   |
|--------|---------------------|-------------------|
| Copy   | 4-letter (32)       | 100%              |
| Copy   | 5-letter (500)      | 100%              |
| Copy   | 8-letter (1000)     | 100%              |
| Reverse| 5-letter (500)       | 100%              |
| Reverse| 8-letter (2000)     | 100%              |
| Addition | 75 train          | 68%               |
| Addition | 2-digit (2000)     | 88.7%             |
| Addition | 3-digit (1000)     | 820/2000 (41%)    |
| Markov | 256 / 5000 seqs     | KL ≈ 0.002        |
| Ternary 2-digit | 80 train   | 1/1 (single test) |

Reverse improves from 61% to 100% with enough data. Addition improves from roughly 16% to 68% (75 train) and 88.7% (2-digit, 2000 train). Copy already did well and remains at 100% with 32+ examples. Markov fits the transition distribution well at both 256 and 5000 sequences. The main takeaway is that task complexity alone does not explain generalization; data size matters.

---

## 4. Implementation Notes and Fixes

Several bugs required fixes during the project:

**Markov lm_head.** The hand-designed Markov model initially failed due to a shape mismatch: the transition matrix log-probs are 3×3, but the lm_head expects (vocab_size, n_embd). Fixed by using only the first three embedding dimensions for the output projection.

**8-letter copy and reverse.** The `masked_tgt` loop used `range(SEQ_LEN + 1, block_size)`, which could index past the end of `tgt`. Updated to `range(SEQ_LEN, len(tgt))`.

**Hand-designed copy 8-letter.** Output showed the separator first and dropped the last letter. The key matrix was wired for positions 9–16, but the first prediction occurs at position 8. Adjusted the key mapping to `range(SEQ_LEN, SEQ_LEN + SEQ_LEN)` with `(t - SEQ_LEN)` so position 8 attends to position 0. The hand-designed model should now reach 100%.

**Addition hand model on GPU.** Tensors in the lm_head calibration loop were created on CPU while the inputs were on GPU. Added `device=device` so all tensors stay on the same device.

**3-digit addition OOM.** Training 10k examples on MPS ran out of memory. Added automatic CPU fallback when `TRAIN_SIZE > 6000`. The 10k run was not completed.

**Hand-designed reverse 5-letter and 8-letter.** Still reports 0%; the attention key mapping for variable-length reverse is incorrect. Not fixed. Trained model results are valid.

---

## 5. Conclusions

Initial runs suggested that task complexity largely determines whether trained models generalize. The extension runs show that data volume is also critical. With little data, models tend to memorize; with enough data, they learn the underlying rules. Copy generalizes even with few examples. Reverse and addition need more data, but both can reach high unseen accuracy when trained on hundreds or thousands of examples. These results align with the view that transformers can learn algorithmic behavior when given sufficient data and an appropriate setup.

**Limitations.** Single seed per run; hand-designed reverse for 5 and 8 letters is broken; 3-digit 10k run not completed; ternary 2-digit uses only one test example; models are minimal (1 layer, 1 head).

---

## 6. Reproducibility

Environment: `conda activate comp560`, `cd myNanoGpt/research`.

Base: `generalize_copy.py`, `generalize_reverse.py`, `generalize_addition.py`.

Extensions: `generalize_copy_4letter.py 32`, `generalize_copy_5letter.py 500`, `generalize_copy_8letter.py 1000`, `generalize_reverse_5letter.py 500`, `generalize_reverse_8letter.py 2000`, `generalize_addition_moredata.py 75`, `generalize_addition_2digit.py 2000`, `generalize_markov_variable.py 256`, `generalize_markov_variable.py 5000`, `generalize_ternary_addition_2digit.py 80`.

---

## 7. Files

| File | Purpose |
|------|---------|
| model.py | Shared GPT architecture |
| generalize_copy.py | Copy 3-letter |
| generalize_copy_4letter.py | Copy 4-letter |
| generalize_copy_5letter.py | Copy 5-letter |
| generalize_copy_8letter.py | Copy 8-letter |
| generalize_reverse.py | Reverse 3-letter |
| generalize_reverse_5letter.py | Reverse 5-letter |
| generalize_reverse_8letter.py | Reverse 8-letter |
| generalize_addition.py | 1-digit addition |
| generalize_addition_moredata.py | 1-digit (variable train size) |
| generalize_addition_2digit.py | 2-digit addition |
| generalize_addition_3digit.py | 3-digit addition |
| generalize_markov_variable.py | Markov (variable train size) |
| generalize_ternary_addition.py | Ternary 1-digit |
| generalize_ternary_addition_2digit.py | Ternary 2-digit |
| compare_training_data.py | Sweep script for train size comparison |
