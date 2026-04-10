# Supervised Fine-Tuning (SFT)

## How It Works

At its core, SFT is teaching the model: "given this input, produce this output."

The model already knows language from pre-training. SFT steers it toward a specific behavior — follow instructions, write code, answer questions in a certain style — by showing it examples and doing gradient descent.

The loss function is **cross-entropy**: for every token in the target response, measure how surprised the model was by it, and adjust weights to be less surprised next time. That's the whole thing.

---

## What You Need

### 1. A Dataset

Each example is a conversation: a sequence of `system` + `user` + `assistant` messages.

```
system:    "You are a Python engineer..."
user:      "Write a function that does X"
assistant: "def do_x(): ..."   ← this is what the model learns to produce
```

The model only computes loss on the **assistant turns**. The system/user messages are context — fed in but not trained on. This is controlled by **weights**: 1.0 on tokens you want to learn, 0.0 on everything else.

### 2. A Base Model

A pre-trained model as the starting point. The model already understands language — SFT just redirects it.

### 3. LoRA (Parameter-Efficient Fine-Tuning)

You don't update all billions of parameters. LoRA adds small trainable matrices on top of the frozen base model. `rank=32` means you're training a tiny fraction of the total parameters.

- Much cheaper (less memory, less compute)
- Faster
- Less prone to catastrophic forgetting

Use ~10x higher LR than you would for full fine-tuning — LoRA needs it.

### 4. A Renderer

Converts human-readable messages into tokens the model understands. Different models have different chat templates (Llama3 formats messages differently than Qwen). The renderer handles this:

- `build_supervised_example(messages)` → tokenizes + sets loss weights
- `build_generation_prompt(messages)` → formats a prompt for inference

### 5. The Training Loop

Each step:
1. **Forward + backward pass** — compute loss, compute gradients
2. **Optimizer step** — apply gradients, update LoRA weights

---

## The Data Pipeline

```
Raw example
    ↓
build_messages()  → list of {role, content} dicts
    ↓
renderer.build_supervised_example()  → (ModelInput, weights)
    ↓
Datum(
    model_input = tokens[:-1],       ← input: all tokens except last
    loss_fn_inputs = {
        target_tokens = tokens[1:],  ← target: shifted by 1
        weights = weights[1:]        ← only loss on assistant tokens
    }
)
```

Standard next-token prediction: input is tokens 0..N-1, target is tokens 1..N. The weights mask out everything except assistant response tokens.

---

## The Training Loop in Detail

### Step 1: Raw Example → Datum

You start with a raw example from the dataset. You convert it into a `Datum`, which has two parts:

```
Datum(
    model_input  = tokens[:-1]     ← what the model sees as input
    loss_fn_inputs = {
        target_tokens = tokens[1:] ← what the model should predict
        weights       = weights[1:]← which tokens to actually compute loss on
    }
)
```

The key insight: **input and target are the same sequence, just shifted by one token.** The model sees token 0..N-1 and tries to predict token 1..N. The `weights` are 0 on system/user tokens and 1 on assistant tokens — so the model is only graded on the assistant response.

### Step 2: Forward Pass

The model takes `model_input` and for each position produces a **probability distribution over the entire vocabulary** — essentially saying "given everything before this token, here's how likely each possible next token is." This gives you logprobs for every token position.

### Step 3: Loss Computation

Cross-entropy loss. For each assistant token position:

```
loss = -logprob(actual_next_token)
```

If the model assigned high probability to the correct token → low loss. If the model was surprised → high loss. Then average across all assistant tokens (weights mask out system/user positions).

### Step 4: Backward Pass (Gradients)

Backpropagation. Given the loss, compute how much each LoRA weight contributed to it. This gives you a **gradient** — a direction in weight space that says "move this weight this way to reduce the loss."

### Step 5: Optimizer Step

Adam takes the gradients and updates the LoRA weights:

```
weight = weight - learning_rate * gradient
```

Adam is smarter than raw gradient descent — it tracks momentum and adapts the step size per parameter. But the core idea is the same: nudge the weights to make the model less surprised by the training data.

Repeat thousands of times. The model's weights slowly shift to assign higher probability to the correct assistant responses.

### One Cycle, Visualized

```
raw example
    │
    ▼
Datum (tokens, weights)
    │
    ▼
model_input → [LLM] → logprobs for each position
                            │
                target_tokens + weights
                            │
                            ▼
                        cross-entropy loss
                            │
                            ▼
                        gradients (backprop)
                            │
                            ▼
                    optimizer updates LoRA weights
```

### Note on Tinker's Async API

`forward_backward` and `optim_step` are two separate calls, dispatched async to the GPU server:

```python
fwd = training_client.forward_backward(training_data, loss_fn="cross_entropy")
opt = training_client.optim_step(AdamParams(learning_rate=1e-4))
fwd.result()
opt.result()
```

Call both before waiting on either — never call them sequentially or you waste GPU time.

---

## Key Hyperparameters

| Param | What it does | Typical range |
|---|---|---|
| `learning_rate` | How big each update step is | `1e-5` to `5e-4` |
| `lora_rank` | Capacity of the LoRA adapter | `16` to `128` |
| `batch_size` | Examples per gradient step | `8` to `128` |

---

## What the Loss Means

Loss starts high and drops as the model learns. In the MBPP experiment: 1.58 → 0.04 over 50 steps on 200 examples.

Near-zero loss on a small dataset means the model has mostly memorized. For real training you need more data, more steps, and eval on a held-out set to check generalization.

---

## The Core Limitation

SFT can only teach the model to **imitate** examples in your dataset. It can't:
- Discover solutions not in the training data
- Optimize for correctness directly
- Learn from its own mistakes

This is why RL, DPO, and distillation exist — they push past imitation.

---

## Implementation

- Notebook: [`methods/sft/sft_train.ipynb`](../methods/sft/sft_train.ipynb)
- Reference: [`tinker-cookbook/tinker_cookbook/recipes/chat_sl/`](../tinker-cookbook/tinker_cookbook/recipes/chat_sl/)
