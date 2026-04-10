# Supervised Fine-Tuning (SFT)

SFT is the simplest form of post-training. Show the model examples of the behavior you want, train with cross-entropy loss, done. It's the baseline everything else is compared against.

Full explanation: [`docs/sft.md`](../../docs/sft.md)

---

## Notebooks

### `sft_train.ipynb`
Original SFT experiment on MBPP (Python coding problems). Trains Llama-3.2-1B to write code given a problem description and test cases.

### `sft_gsm8k_cot.ipynb`
SFT on GSM8K using **full chain-of-thought reasoning**. The model learns to reason step by step before producing the final answer.

```
Q: Janet's ducks lay 16 eggs per day...
A: Janet sells 16 - 3 - 4 = 9 duck eggs per day.
   She makes 9 * 2 = 18 dollars per day.
   #### 18
```

### `sft_gsm8k_answer_only.ipynb`
SFT on GSM8K using **only the final answer**. The model learns to produce the correct number with no reasoning.

```
Q: Janet's ducks lay 16 eggs per day...
A: 18
```

---

## CoT vs Answer Only

The two GSM8K notebooks are designed to be compared directly:

| | CoT | Answer Only |
|---|---|---|
| What model learns | Reasoning + answer | Just the answer |
| Loss drops | Slower (longer sequences) | Faster (1-3 tokens) |
| Generalizes to unseen problems | Better | Worse |
| Useful as base for RL | Yes | No |

Expected result: CoT accuracy > answer-only accuracy on held-out test set, even though answer-only loss drops faster during training.

---

## Config

All three notebooks use the same setup:

| Param | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-1B` |
| LoRA rank | 32 |
| Learning rate | `1e-4` |
| Train examples | 500 (GSM8K) / 200 (MBPP) |
| Steps | 50 |
| Eval examples | 100 |
