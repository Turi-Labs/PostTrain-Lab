# Understanding RL for LLMs: A Step-by-Step Guide

> **Based on the Tinker Cookbook** - This guide explains Reinforcement Learning for Large Language Models in simple terms, drawing from the official Tinker documentation.

## 📚 Table of Contents

1. [What is RL for LLMs?](#what-is-rl-for-llms)
2. [Different RL Methods](#different-rl-methods)
3. [Step-by-Step Pipeline](#step-by-step-pipeline)
4. [Code Walkthrough](#code-walkthrough)
5. [Tinker-Specific Concepts](#tinker-specific-concepts)

---

## What is RL for LLMs?

### The Core Idea

**Reinforcement Learning** = Learning from trial and error

Instead of showing the model correct answers (like in SFT), we:
1. Let the model generate multiple attempts
2. Score each attempt with a reward function
3. Update the model to produce higher-scoring outputs

### Simple Analogy

Think of teaching a dog tricks:

- **Supervised Learning (SFT)**: You physically guide the dog's paw to shake hands every time
- **Reinforcement Learning (RL)**: You say "shake" and give treats when the dog gets it right, ignore when wrong

The dog learns by exploring and getting feedback!

### Why Use RL?

**Use RL when you have:**
- ❌ No perfect solutions to copy
- ✅ A way to score/evaluate outputs
- ✅ Compute budget for exploration
- ✅ A metric you want to optimize

**Example for Code Generation:**
- You have: Test cases ✅
- You don't have: The perfect solution ❌
- You can: Run code and check if tests pass ✅
- **Perfect for RL!**

---

## Different RL Methods

According to Tinker Cookbook, there are **two main types** of RL:

### 1. RL with Verifiable Rewards (RLVR) 🧪

**What it is:**
- Use a program to automatically check if model output is correct
- No human judgment needed
- Binary or computed rewards

**When to use:**
- Code generation (does it pass tests?)
- Math problems (is the answer correct?)
- Tool use (did the task succeed?)

**Example:**
```python
def get_reward(generated_code, test_cases):
    if code_passes_all_tests(generated_code, test_cases):
        return 1.0  # Success!
    else:
        return 0.0  # Failure
```

### 2. RL from Human Feedback (RLHF) 👥

**What it is:**
- Train a "reward model" on human preferences
- Use reward model to score outputs
- Optimize for subjective qualities

**When to use:**
- Helpfulness (is the answer clear?)
- Style (is it polite?)
- Safety (is it harmful?)

**Example:**
```python
# First: Train a reward model on human preference data
reward_model = train_preference_model(human_comparisons)

# Then: Use it to score outputs
def get_reward(generated_text):
    return reward_model.score(generated_text)
```

---

## Step-by-Step Pipeline

### The Complete RL Training Loop

Here's what happens in **every RL iteration**:

```
┌─────────────────────────────────────────────────────────────┐
│                     RL TRAINING LOOP                        │
└─────────────────────────────────────────────────────────────┘

Step 1: SAVE WEIGHTS & CREATE SAMPLER
        ↓
   [Training Client] → Save current weights → [Sampling Client]
        
        Why? We need to sample from the current policy

Step 2: GENERATE ROLLOUTS (Sample multiple solutions)
        ↓
   For each problem:
   ├─ Sample solution #1 (with temperature > 0)
   ├─ Sample solution #2
   ├─ Sample solution #3
   └─ Sample solution #4
   
        Why? Multiple samples give us comparison data

Step 3: COMPUTE REWARDS (Evaluate each solution)
        ↓
   For each solution:
   ├─ Solution #1: Execute tests → reward = 1.0 ✅
   ├─ Solution #2: Execute tests → reward = 0.5 ⚠️
   ├─ Solution #3: Execute tests → reward = 0.0 ❌
   └─ Solution #4: Execute tests → reward = 1.0 ✅
   
        Why? We need feedback on what worked

Step 4: COMPUTE ADVANTAGES (Normalize rewards)
        ↓
   Mean reward = (1.0 + 0.5 + 0.0 + 1.0) / 4 = 0.625
   
   Advantages:
   ├─ Solution #1: 1.0 - 0.625 = +0.375  (better than average)
   ├─ Solution #2: 0.5 - 0.625 = -0.125  (worse than average)
   ├─ Solution #3: 0.0 - 0.625 = -0.625  (much worse)
   └─ Solution #4: 1.0 - 0.625 = +0.375  (better than average)
   
        Why? Positive advantage = reinforce, Negative = discourage

Step 5: PREPARE TRAINING DATA
        ↓
   For each solution, create a Datum:
   ├─ model_input: prompt + generated tokens
   ├─ target_tokens: the tokens we sampled
   ├─ logprobs: log probabilities from sampling
   └─ advantages: computed advantages
   
        Why? This is what Tinker needs for training

Step 6: TRAINING UPDATE (Policy Gradient)
        ↓
   training_client.forward_backward(data, loss_fn="ppo")
   training_client.optim_step(AdamParams(lr=5e-5))
   
        Why? Update model to favor high-advantage solutions

Step 7: REPEAT
        ↓
   Go back to Step 1 with updated model
```

### Key Concepts Explained

#### 🎲 Rollouts
**What:** A complete episode where the model generates a solution

**Contains:**
- The prompt (problem description)
- Generated tokens (model's solution)
- Log probabilities (how confident was the model?)
- Reward (how good was the solution?)

#### 🎯 Rewards
**What:** Numerical feedback on how good the output was

**Range:** Typically -1.0 to +1.0

**Examples:**
- Code passes all tests: **+1.0**
- Code passes some tests: **+0.5**
- Code fails all tests: **-0.5**
- Syntax error: **-1.0**

#### ⚖️ Advantages
**What:** Relative quality compared to other attempts

**Formula:** `advantage = reward - mean_reward_in_group`

**Why:** Helps the model understand "this was better than my usual attempt"

**Important:** Advantages are computed **per problem group**
- If all attempts get reward 0.5, all advantages are 0 (no learning signal)
- If attempts get 0.0, 0.5, 1.0, advantages are -0.5, 0.0, +0.5 (clear signal)

---

## Code Walkthrough

### Minimal RL Loop (from Tinker Cookbook)

Let me break down the `rl_loop.py` from the cookbook:

```python
# ═══════════════════════════════════════════════════════
# STEP 1: SETUP
# ═══════════════════════════════════════════════════════

# Get tokenizer and renderer (converts messages ↔ tokens)
tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
renderer = renderers.get_renderer("llama3", tokenizer)

# Load dataset (e.g., math problems)
dataset = load_dataset("openai/gsm8k", "main")

# Create training client (this is your model)
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.1-8B",
    rank=32  # LoRA rank
)
```

```python
# ═══════════════════════════════════════════════════════
# STEP 2: MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════

for batch_idx in range(num_batches):
    
    # 2a. Create sampler from current weights
    sampling_path = training_client.save_weights_for_sampler(f"batch_{batch_idx}").result().path
    sampling_client = service_client.create_sampling_client(model_path=sampling_path)
    
    # Why? We need to sample from the CURRENT policy, not the old one
```

```python
    # 2b. Generate rollouts for each problem
    for problem in batch:
        # Build prompt
        prompt = renderer.build_generation_prompt([
            {"role": "user", "content": problem["question"]}
        ])
        
        # Sample MULTIPLE solutions (this is key for RL!)
        sample_result = sampling_client.sample(
            prompt=prompt,
            num_samples=16,  # Generate 16 different solutions
            sampling_params=SamplingParams(
                max_tokens=256,
                temperature=0.8  # Higher = more diverse
            )
        ).result()
```

```python
        # 2c. Compute rewards for each solution
        rewards = []
        for sequence in sample_result.sequences:
            # Extract the answer
            response_text = renderer.parse_response(sequence.tokens)[0]["content"]
            
            # Check if correct
            is_correct = check_answer(response_text, problem["answer"])
            reward = 1.0 if is_correct else 0.0
            
            rewards.append(reward)
```

```python
        # 2d. Compute advantages (center around mean)
        mean_reward = sum(rewards) / len(rewards)
        advantages = [r - mean_reward for r in rewards]
        
        # Example:
        # rewards = [1.0, 1.0, 0.0, 0.0]
        # mean_reward = 0.5
        # advantages = [+0.5, +0.5, -0.5, -0.5]
```

```python
        # 2e. Create training data
        datums = []
        for sequence, advantage in zip(sample_result.sequences, advantages):
            # Combine prompt + completion
            full_tokens = prompt.to_ints() + sequence.tokens
            
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(full_tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": full_tokens[1:],
                    "logprobs": sequence.logprobs,
                    "advantages": [advantage] * len(sequence.tokens)
                }
            )
            datums.append(datum)
```

```python
    # 2f. Train the model
    fwd_bwd = training_client.forward_backward(
        datums,
        loss_fn="importance_sampling"  # or "ppo"
    )
    optim_step = training_client.optim_step(
        AdamParams(learning_rate=4e-5)
    )
    
    # Wait for results
    fwd_bwd.result()
    optim_step.result()
    
    # The model is now updated! Loop continues...
```

---

## Tinker-Specific Concepts

### 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                    YOUR COMPUTER                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Your Python Script                              │ │
│  │  - Loads data                                    │ │
│  │  - Calls Tinker API                              │ │
│  │  - Computes rewards                              │ │
│  │  - Logs metrics                                  │ │
│  └─────────────┬────────────────────────────────────┘ │
└────────────────┼───────────────────────────────────────┘
                 │
                 │ API Calls
                 ↓
┌────────────────────────────────────────────────────────┐
│                  TINKER SERVERS                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  GPU Cluster                                     │ │
│  │  - Stores model weights                          │ │
│  │  - Runs forward/backward passes                  │ │
│  │  - Runs sampling                                 │ │
│  │  - Updates weights                               │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

**Key Point:** Tinker does the heavy GPU work; you handle the logic!

### 📦 Key Data Types

#### `Datum` - A single training example

```python
datum = types.Datum(
    model_input=ModelInput.from_ints([123, 456, 789]),  # Input tokens
    loss_fn_inputs={
        "target_tokens": [456, 789, 101],      # What to predict
        "logprobs": [-0.5, -0.3, -0.8],        # From sampling
        "advantages": [0.2, 0.2, 0.2]          # From reward
    }
)
```

#### `ModelInput` - Token sequence (can include images!)

```python
# Text only
model_input = ModelInput.from_ints([1, 2, 3, 4, 5])

# With images (for VLMs)
model_input = ModelInput(chunks=[
    EncodedTextChunk(tokens=[1, 2, 3]),
    ImageChunk(data=image_bytes, format="png"),
    EncodedTextChunk(tokens=[4, 5, 6])
])
```

#### `Renderer` - Converts messages ↔ tokens

```python
# Build prompt for generation
prompt = renderer.build_generation_prompt([
    {"role": "user", "content": "Hello!"}
])

# Parse generated tokens back to message
message, success = renderer.parse_response(tokens)
# message = {"role": "assistant", "content": "Hi there!"}
```

### 🎛️ Configuration Patterns

Tinker uses a naming convention for dimensions:

- `_P`: **Problems** - Different questions/prompts in a batch
- `_G`: **Groups** - Multiple rollouts per problem
- `_T`: **Tokens** - Sequence positions
- `_D`: **Datums** - Training examples (flattened P × G)

Example:
```python
rewards_P_G: list[list[float]]  # rewards_P_G[problem][group]
tokens_G_T: list[list[int]]     # tokens_G_T[group][token]
```

### 🔄 The Group Concept

**Why groups?**

Groups allow us to compare multiple attempts at the same problem:

```
Problem: "Write a function to add two numbers"
├─ Group member 1: def add(a,b): return a+b     → reward: 1.0
├─ Group member 2: def add(a,b): return a-b     → reward: 0.0
├─ Group member 3: def add(a,b): return a*b     → reward: 0.0
└─ Group member 4: def add(a,b): return a+b     → reward: 1.0

Mean reward: 0.5
Advantages: [+0.5, -0.5, -0.5, +0.5]
```

**Key insight:** Advantages are centered **within each group**
- This makes the model prefer solutions that are better than its current average
- Prevents reward drift and instability

### 📊 Loss Functions

Tinker provides several RL loss functions:

#### `importance_sampling` - Basic REINFORCE

```python
loss = -log_prob(action) * advantage
```

**Use when:** Simple RL, on-policy training

#### `ppo` - Proximal Policy Optimization

```python
ratio = new_policy / old_policy
clipped_ratio = clip(ratio, 0.8, 1.2)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

**Use when:** Want stability, off-policy training

**Hyperparameters:**
```python
loss_fn_config={
    "clip_low_threshold": 0.8,   # Don't decrease prob too much
    "clip_high_threshold": 1.2   # Don't increase prob too much
}
```

#### `cispo` - Clipped Importance Sampling

```python
clipped_ratio = clip(new_policy / old_policy, 0.8, 1.2)
loss = -clipped_ratio * log_prob(action) * advantage
```

**Use when:** Want something between importance sampling and PPO

#### `dro` - Direct Reward Optimization

```python
loss = log_prob * advantage - 0.5 * beta * (log(new_policy / old_policy))^2
```

**Use when:** Offline RL, want quadratic penalty on policy change

### 🔧 Hyperparameters Guide

| Parameter | Typical Value | Range | What it does |
|-----------|--------------|-------|--------------|
| **learning_rate** | 4e-5 | 1e-5 - 1e-4 | How fast to update weights |
| **temperature** | 0.8 | 0.6 - 1.0 | Sampling diversity (higher = more random) |
| **group_size** | 4-16 | 2 - 32 | Solutions per problem (more = better estimates) |
| **batch_size** | 32-128 | 16 - 256 | Problems per batch (more = more stable) |
| **lora_rank** | 32 | 8 - 64 | LoRA adaptation size |
| **max_tokens** | 256-512 | 128 - 2048 | Max length of generated responses |

---

## Different Ways to Train an LLM with RL

Based on the Tinker Cookbook, here are the main approaches:

### 1. **Single-Turn RL** (Simple)

Model generates ONE response, gets reward, updates

```python
Question → Model → Answer → Reward → Update
```

**Examples:**
- Math problems (GSM8K)
- Code generation (MBPP, HumanEval)
- Classification

### 2. **Multi-Turn RL** (Advanced)

Model has a conversation, gets reward at the end

```python
Q1 → A1 → Q2 → A2 → Q3 → A3 → Final Reward → Update
```

**Examples:**
- Interactive debugging
- Tool use (search, calculator)
- Multi-step reasoning

### 3. **Multi-Agent RL** (Research)

Multiple models interact, each gets rewards

```python
Agent 1 ↔ Agent 2 ↔ Environment
  ↓         ↓
Reward1   Reward2
```

**Examples:**
- Debate
- Collaborative problem solving
- Game playing

### 4. **Offline RL** (Data-Efficient)

Train on pre-collected rollouts (no new sampling)

```python
Load old rollouts → Compute rewards → Train with DRO/CQL
```

**Examples:**
- When sampling is expensive
- When you have lots of existing data
- Safety-critical applications

---

## Summary: The Big Picture

### What RL Training Does

1. **Exploration**: Model tries different solutions
2. **Evaluation**: Reward function scores each attempt
3. **Learning**: Model updates to favor high-reward solutions
4. **Iteration**: Repeat many times to improve

### Key Differences from SFT

| Aspect | SFT | RL |
|--------|-----|-----|
| **Data** | Need correct answers | Need scoring function |
| **Learning** | Mimic examples | Maximize rewards |
| **Exploration** | None | Essential |
| **Compute** | Low | Medium-High |
| **Flexibility** | Limited to training data | Can discover novel solutions |

### When to Use What

```
Start with base model (e.g., Llama-3.2-1B)
         ↓
    Do you have correct solutions?
         ↓
   YES         NO
    ↓           ↓
   SFT        RLVR
    ↓           ↓
Want even better?
    ↓
   SFT → RLVR (Best!)
```

---

## Next Steps

1. **Read:** The code in `methods/rlvr/rlvr_train.py`
2. **Run:** `python methods/rlvr/rlvr_train.py` (start small!)
3. **Experiment:** Change hyperparameters and see what happens
4. **Explore:** Look at Tinker Cookbook recipes (`tinker-cookbook/recipes/`)
5. **Build:** Create your own reward function for your domain

## Further Reading

- **Tinker Docs:** `/tinker-cookbook/docs/rl/`
- **Recipes:** `/tinker-cookbook/tinker_cookbook/recipes/`
  - `rl_basic.py` - Minimal RL example
  - `rl_loop.py` - Simple training loop
  - `code_rl/` - Code generation RL
  - `math_rl/` - Math problem RL
  - `verifiers_rl/` - RL with outcome verifiers

---

**Questions?** Read this guide again slowly, then look at the code! 🚀

