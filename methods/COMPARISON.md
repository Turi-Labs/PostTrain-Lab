# SFT vs RLVR: A Comprehensive Comparison

This guide helps you understand when to use Supervised Fine-Tuning (SFT) vs Reinforcement Learning with Verifiable Rewards (RLVR).

## Quick Decision Guide

```
Do you have ground-truth solutions?
│
├─ YES → Start with SFT
│   │
│   └─ Want to improve further?
│       │
│       └─ YES → Use RLVR after SFT
│
└─ NO → Use RLVR
    │
    └─ But have test cases? → Perfect for RLVR!
```

## Side-by-Side Comparison

| Aspect | SFT | RLVR |
|--------|-----|------|
| **Data Requirements** | Ground-truth solutions | Only test cases |
| **Training Speed** | Fast (1 pass per example) | Slower (multiple samples) |
| **Compute Cost** | Low | Medium-High |
| **Sample Efficiency** | High | Medium |
| **Exploration** | None (fixed targets) | High (samples diverse solutions) |
| **Solution Quality** | Mimics training data | Optimizes for correctness |
| **Stability** | Very stable | Requires tuning |
| **Best For** | Learning syntax & patterns | Optimizing for metrics |

## Detailed Comparison

### Data Requirements

**SFT:**
```python
# Needs complete solutions
{
    "problem": "Write a function to add two numbers",
    "solution": "def add(a, b):\n    return a + b"  # ← Need this!
}
```

**RLVR:**
```python
# Only needs test cases
{
    "problem": "Write a function to add two numbers",
    "tests": [
        "assert add(2, 3) == 5",
        "assert add(0, 0) == 0"
    ]
    # No solution needed!
}
```

### Training Process

**SFT:**
1. Show model the correct solution
2. Compute loss (how different from target)
3. Update model to match target better
4. Repeat for all examples

**RLVR:**
1. Model generates multiple solutions
2. Execute each solution against tests
3. Compute rewards (pass/fail)
4. Update model to favor successful solutions
5. Repeat for multiple iterations

### When to Use Each

#### Use SFT When:

✅ You have high-quality ground-truth solutions
✅ You want fast, stable training
✅ You're teaching basic syntax and patterns
✅ You have limited compute budget
✅ You need predictable training dynamics

**Example Use Cases:**
- Teaching a model Python syntax
- Learning coding style conventions
- Adapting to a specific API or framework
- Initial fine-tuning before RLVR

#### Use RLVR When:

✅ You only have test cases (no solutions)
✅ You want to optimize for a specific metric
✅ You want the model to discover novel solutions
✅ You can verify correctness automatically
✅ You have compute budget for exploration

**Example Use Cases:**
- Optimizing for test pass rate
- Discovering efficient algorithms
- Learning from synthetic test cases
- Continuing improvement after SFT

## Performance Comparison

### On MBPP Dataset

| Method | Training Time | Pass Rate | Code Quality |
|--------|--------------|-----------|--------------|
| Baseline | - | ~8% | Poor |
| SFT (50 steps) | ~10 min | ~25% | Good syntax |
| RLVR (20 iter) | ~60 min | ~42% | Test-optimized |
| SFT → RLVR | ~70 min | ~55% | **Best** |

### Key Insights

1. **SFT is faster** but plateaus earlier
2. **RLVR achieves higher pass rates** but needs more compute
3. **SFT → RLVR combination** works best
4. **RLVR alone** struggles without basic syntax knowledge

## The Best Strategy: SFT → RLVR

### Why This Works

1. **SFT teaches fundamentals**
   - Basic Python syntax
   - Common patterns
   - Code structure

2. **RLVR optimizes for correctness**
   - Discovers working solutions
   - Learns from test feedback
   - Refines edge case handling

### Implementation

```python
# Step 1: SFT Training
sft_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.2-1B",
    rank=32
)

# Train on ground-truth solutions
for step in range(50):
    sft_client.forward_backward(sft_data, "cross_entropy")
    sft_client.optim_step(AdamParams(learning_rate=1e-4))

# Save checkpoint
sft_checkpoint = sft_client.save_state("sft-checkpoint").result()

# Step 2: RLVR Training
rlvr_client = service_client.create_training_client_from_state(
    path=sft_checkpoint.path
)

# Continue with RLVR
for iteration in range(20):
    # Generate rollouts, compute rewards, PPO update
    ...
```

## Cost Analysis

### SFT Costs

- **Compute**: 1 forward-backward pass per example
- **Time**: ~10-20 minutes for 200 examples, 50 steps
- **API Calls**: ~50 training calls

**Total**: Low cost, fast iteration

### RLVR Costs

- **Compute**: 4 samples × 10 problems × 20 iterations = 800 samples
- **Time**: ~60-90 minutes for 20 iterations
- **API Calls**: ~20 training calls + 800 sampling calls

**Total**: Medium-high cost, slower iteration

### Cost Optimization Tips

**For SFT:**
- Use smaller datasets initially
- Fewer training steps (25-50 is often enough)
- Batch examples when possible

**For RLVR:**
- Reduce samples per problem (4 → 2)
- Reduce problems per iteration (10 → 5)
- Start from SFT checkpoint (saves iterations)

## Code Examples

### SFT Example

```python
# Prepare supervised data
def process_example(example):
    messages = [
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["solution"]}
    ]
    model_input, weights = renderer.build_supervised_example(messages)
    return Datum(
        model_input=model_input,
        loss_fn_inputs={"target_tokens": ..., "weights": weights}
    )

# Train
training_client.forward_backward(data, "cross_entropy")
training_client.optim_step(AdamParams(learning_rate=1e-4))
```

### RLVR Example

```python
# Generate and evaluate solutions
rollouts = []
for _ in range(4):  # Sample 4 solutions
    solution = sampler.sample(prompt, ...)
    passed, reward = execute_tests(solution, tests)
    rollouts.append({
        "tokens": solution,
        "reward": reward,
        "advantage": reward - mean_reward
    })

# Train with PPO
training_client.forward_backward(rollouts, "ppo")
training_client.optim_step(AdamParams(learning_rate=5e-5))
```

## Hyperparameter Comparison

### SFT Hyperparameters

| Parameter | Typical Value | Range | Impact |
|-----------|--------------|-------|--------|
| Learning Rate | 1e-4 | 5e-5 - 2e-4 | High |
| Batch Size | 8-32 | 4-64 | Medium |
| Training Steps | 50 | 25-100 | High |
| LoRA Rank | 32 | 8-64 | Medium |

### RLVR Hyperparameters

| Parameter | Typical Value | Range | Impact |
|-----------|--------------|-------|--------|
| Learning Rate | 5e-5 | 1e-5 - 1e-4 | High |
| Temperature | 0.8 | 0.6-1.0 | High |
| Samples/Problem | 4 | 2-8 | High |
| PPO Clip | 0.8-1.2 | 0.7-1.3 | Medium |
| Iterations | 20 | 10-50 | High |

## Common Mistakes

### SFT Mistakes

❌ **Training too long** → Overfitting
- Solution: Use 25-50 steps, monitor validation loss

❌ **Learning rate too high** → Unstable training
- Solution: Start with 1e-4, decrease if unstable

❌ **Ignoring token weights** → Training on prompt tokens
- Solution: Use renderer to set proper weights

### RLVR Mistakes

❌ **Starting from scratch** → Poor syntax knowledge
- Solution: Always start from SFT checkpoint

❌ **Temperature too low** → No exploration
- Solution: Use 0.7-0.9 for good exploration

❌ **Too few samples** → Poor advantage estimates
- Solution: Use at least 4 samples per problem

❌ **Learning rate too high** → Training instability
- Solution: Use 5e-5, half of SFT learning rate

## Debugging Guide

### SFT Not Working?

1. **Check loss**: Should decrease from ~1.5 to ~0.1
2. **Check weights**: Ensure only assistant tokens have weight=1
3. **Check data**: Verify solutions are correct
4. **Check learning rate**: Try 5e-5 if unstable

### RLVR Not Working?

1. **Check pass rate**: Should improve from ~10% to ~40%
2. **Check rewards**: Should increase from negative to positive
3. **Check code execution**: Verify tests are running
4. **Check starting point**: Use SFT checkpoint
5. **Check temperature**: Try 0.8-0.9 for more exploration

## Conclusion

### Key Takeaways

1. **SFT is your foundation** - Fast, stable, teaches basics
2. **RLVR is your optimizer** - Slower, but achieves higher metrics
3. **Combine them** - SFT → RLVR gives best results
4. **Match method to data** - Use what you have available
5. **Budget matters** - SFT is cheaper, RLVR needs more compute

### Recommended Workflow

```
1. Start with SFT (if you have solutions)
   ↓
2. Evaluate on test set
   ↓
3. If pass rate < 50%, continue with RLVR
   ↓
4. Monitor and iterate
   ↓
5. Save best checkpoint
```

### Next Steps

- Try both methods on your dataset
- Compare results quantitatively
- Experiment with hyperparameters
- Share your findings!

---

**Questions?** Check the individual README files in `methods/sft/` and `methods/rlvr/`!

