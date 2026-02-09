# RL for LLMs: Quick Reference Card

## 🎯 The Core Loop (5 Steps)

```
1. SAMPLE    → Generate multiple solutions per problem
2. EVALUATE  → Run reward function on each solution  
3. ADVANTAGE → Compare: reward - mean(group_rewards)
4. PREPARE   → Create Datum objects with advantages
5. TRAIN     → forward_backward() + optim_step()
```

## 📊 Key Formulas

### Advantage Computation
```python
advantages = [reward - mean_reward for reward in rewards]
```

### Loss Functions

**Importance Sampling:**
```python
loss = -log_prob(tokens) * advantage
```

**PPO:**
```python
ratio = exp(new_logprob - old_logprob)
clipped = clip(ratio, 0.8, 1.2)
loss = -min(ratio * advantage, clipped * advantage)
```

## 🔧 Essential Code Patterns

### 1. Create Sampler from Training Client

```python
# Save current weights
sampling_path = training_client.save_weights_for_sampler("iter_0").result().path

# Create sampler
sampling_client = service_client.create_sampling_client(model_path=sampling_path)
```

### 2. Generate Rollouts

```python
# Sample multiple solutions
result = sampling_client.sample(
    prompt=prompt,
    num_samples=4,  # Group size
    sampling_params=types.SamplingParams(
        max_tokens=400,
        temperature=0.8,  # Higher = more diverse
        stop=renderer.get_stop_sequences()
    )
).result()

# Extract tokens and logprobs
for seq in result.sequences:
    tokens = seq.tokens
    logprobs = seq.logprobs  # For importance sampling
```

### 3. Compute Rewards

```python
rewards = []
for seq in result.sequences:
    # Parse response
    response, _ = renderer.parse_response(seq.tokens)
    code = response["content"]
    
    # Execute and evaluate
    passed, num_passed, error = execute_tests(code, test_cases)
    reward = 1.0 if passed else 0.0
    
    rewards.append(reward)
```

### 4. Compute Advantages

```python
# Group normalization (per problem)
mean_reward = sum(rewards) / len(rewards)
advantages = [r - mean_reward for r in rewards]
```

### 5. Create Training Data

```python
datums = []
for seq, advantage in zip(result.sequences, advantages):
    # Combine prompt + completion
    full_tokens = prompt.to_ints() + seq.tokens
    
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(full_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": full_tokens[1:],
            "logprobs": seq.logprobs,
            "advantages": [advantage] * len(seq.tokens)
        }
    )
    datums.append(datum)
```

### 6. Training Update

```python
# Forward-backward with PPO
fwd_bwd = training_client.forward_backward(
    datums,
    loss_fn="ppo",
    loss_fn_config={
        "clip_low_threshold": 0.8,
        "clip_high_threshold": 1.2
    }
)

# Optimizer step
optim = training_client.optim_step(
    types.AdamParams(learning_rate=5e-5)
)

# Wait for results
fwd_bwd.result()
optim.result()
```

## ⚙️ Hyperparameters Cheat Sheet

| What | Value | Why |
|------|-------|-----|
| **Learning Rate** | 5e-5 | Half of SFT (RL is less stable) |
| **Temperature** | 0.7-0.9 | High enough for exploration |
| **Group Size** | 4-8 | Balance compute vs variance |
| **Batch Size** | 16-32 problems | Enough for stable gradients |
| **PPO Clip** | 0.8-1.2 | Prevent large policy changes |
| **Max Tokens** | 400-600 | Long enough for solutions |

## 🚨 Common Mistakes

### ❌ Mistake 1: Not creating new sampler

```python
# WRONG - uses old policy
sampler = training_client.save_weights_and_get_sampling_client("once")
for iteration in range(10):
    rollouts = sample(sampler)  # ❌ Always same policy!
```

```python
# CORRECT - creates new sampler each iteration
for iteration in range(10):
    sampler = training_client.save_weights_and_get_sampling_client(f"iter_{iteration}")
    rollouts = sample(sampler)  # ✅ Updated policy
```

### ❌ Mistake 2: Temperature too low

```python
# WRONG - no exploration
sampling_params = SamplingParams(temperature=0.2)  # ❌ Too greedy
```

```python
# CORRECT - good exploration
sampling_params = SamplingParams(temperature=0.8)  # ✅ Diverse samples
```

### ❌ Mistake 3: Forgetting to center advantages

```python
# WRONG - raw rewards as advantages
advantages = rewards  # ❌ No comparison
```

```python
# CORRECT - center within group
mean_reward = sum(rewards) / len(rewards)
advantages = [r - mean_reward for r in rewards]  # ✅ Relative quality
```

### ❌ Mistake 4: Wrong advantage shape

```python
# WRONG - single advantage value
loss_fn_inputs={
    "advantages": advantage  # ❌ Should be a list
}
```

```python
# CORRECT - advantage per token
loss_fn_inputs={
    "advantages": [advantage] * len(tokens)  # ✅ Per-token
}
```

## 📈 What to Expect

### Typical Learning Curve

```
Iteration 1-5:    Low rewards, lots of errors
Iteration 6-10:   Some successes appearing
Iteration 11-15:  Steady improvement
Iteration 16-20:  Plateauing (may need more data/steps)
```

### Good Signs
- ✅ Mean reward increasing
- ✅ Pass rate improving
- ✅ Entropy decreasing slowly
- ✅ KL divergence staying small (<0.1)

### Bad Signs
- ❌ Rewards not changing (temperature too low?)
- ❌ Loss exploding (learning rate too high?)
- ❌ KL divergence large (>1.0, policy diverged)
- ❌ All advantages zero (rewards identical)

## 🎓 Tinker-Specific Tips

### Async Patterns

```python
# Submit both operations before waiting
fwd_bwd_future = await training_client.forward_backward_async(data, "ppo")
optim_future = await training_client.optim_step_async(adam_params)

# Now wait
fwd_bwd_result = await fwd_bwd_future
optim_result = await optim_future
```

### Renderers

```python
# Get recommended renderer for model
renderer_name = model_info.get_recommended_renderer_name(model_name)
renderer = renderers.get_renderer(renderer_name, tokenizer)

# Build prompt
prompt = renderer.build_generation_prompt(messages)

# Parse response
response, success = renderer.parse_response(tokens)
```

### Checkpointing

```python
# Save for resuming training (includes optimizer state)
training_client.save_state("checkpoint_10").result()

# Save for sampling only (faster, smaller)
training_client.save_weights_for_sampler("sampler_10").result()

# Resume training
training_client = service_client.create_training_client_from_state_with_optimizer(
    path="tinker://your-checkpoint-path"
)
```

## 🔍 Debugging Checklist

Problem not improving?

- [ ] Is temperature high enough? (try 0.8)
- [ ] Are you creating new sampler each iteration?
- [ ] Are advantages being computed correctly?
- [ ] Is learning rate appropriate? (try 5e-5)
- [ ] Are group sizes large enough? (try 4-8)
- [ ] Is reward function working? (print some examples)
- [ ] Is model strong enough? (try starting from SFT)

## 📚 More Resources

- **Full Guide:** `docs/RL_EXPLAINED.md`
- **Implementation:** `methods/rlvr/rlvr_train.py`
- **Tinker Docs:** `tinker-cookbook/docs/rl/`
- **Example Recipes:** `tinker-cookbook/tinker_cookbook/recipes/`

---

**Pro Tip:** Start small! Use 2-3 iterations, 2 problems, 2 samples per problem to test your pipeline. Then scale up.

