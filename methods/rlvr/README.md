# RLVR: Reinforcement Learning with Verifiable Rewards

## Overview

RLVR (Reinforcement Learning with Verifiable Rewards) is a training approach that learns from automated execution feedback rather than requiring ground-truth solutions. This is particularly powerful for code generation tasks where we can verify correctness by running test cases.

## Key Concepts

### How RLVR Works

1. **Sample Multiple Solutions** - Generate diverse candidate solutions using temperature sampling
2. **Execute & Verify** - Run each solution against test cases to get binary pass/fail feedback
3. **Compute Rewards** - Convert test results into reward signals
4. **Policy Gradient Update** - Use PPO to reinforce successful solutions and discourage failures

### Advantages over SFT

| Aspect | SFT | RLVR |
|--------|-----|------|
| **Data Requirements** | Needs ground-truth solutions | Only needs test cases |
| **Exploration** | Limited to training examples | Discovers novel solutions |
| **Optimization Target** | Mimics human solutions | Directly optimizes for correctness |
| **Scalability** | Limited by human annotations | Can scale with synthetic tests |

## Implementation Details

### Reward Function

```python
def compute_reward(all_passed: bool, num_passed: int, total_tests: int) -> float:
    if all_passed:
        return 1.0  # Perfect solution
    elif num_passed > 0:
        return 0.5 * (num_passed / total_tests)  # Partial credit
    else:
        return -0.5  # Complete failure
```

This reward structure:
- Strongly rewards fully correct solutions (+1.0)
- Gives partial credit for partially working code
- Penalizes completely broken solutions (-0.5)

### Advantage Computation

We use **group-based advantage normalization**:

```python
advantage = reward - mean(rewards_in_group)
```

This means:
- Solutions better than average get positive advantages (reinforced)
- Solutions worse than average get negative advantages (discouraged)
- The model learns to prefer better solutions within each problem

### PPO Loss

We use Proximal Policy Optimization (PPO) to prevent destructive updates:

```python
loss_fn="ppo"
loss_fn_config={
    "clip_low_threshold": 0.8,   # Don't decrease probability too much
    "clip_high_threshold": 1.2   # Don't increase probability too much
}
```

This ensures the model doesn't deviate too far from the sampling policy, maintaining training stability.

## Usage

### Basic Training

```bash
python rlvr_train.py
```

### Configuration

Edit the config dictionary in `rlvr_train.py`:

```python
config = {
    "base_model": "meta-llama/Llama-3.2-1B",
    "lora_rank": 32,
    "num_train_problems": 100,
    "num_iterations": 20,
    "problems_per_iteration": 10,
    "samples_per_problem": 4,
    "learning_rate": 5e-5,
    "temperature": 0.8,
    "max_tokens": 400,
    "ppo_epochs": 1,
    "ppo_clip_low": 0.8,
    "ppo_clip_high": 1.2,
}
```

### Starting from SFT Checkpoint

You can initialize RLVR from a supervised fine-tuned model:

```python
config = {
    ...
    "start_from_checkpoint": "tinker://your-sft-checkpoint-path",
}
```

This often works better than starting from scratch!

## Training Process

### Iteration Flow

Each training iteration:

1. **Sample Problems** - Randomly select N problems from training set
2. **Generate Rollouts** - Sample M solutions per problem
3. **Execute Code** - Run each solution against test cases
4. **Compute Rewards** - Convert test results to rewards
5. **Normalize Advantages** - Center advantages per problem group
6. **PPO Update** - Update model to favor better solutions

### Expected Behavior

You should see:
- **Mean reward** gradually increasing (from ~0 to ~0.5+)
- **Pass rate** improving (from ~10% to ~40%+)
- **Loss** decreasing initially, then stabilizing

### Typical Timeline

- **Iterations 1-5**: Model explores, many failures
- **Iterations 6-10**: Model starts finding working solutions
- **Iterations 11-20**: Model refines and improves success rate

## Hyperparameter Guide

### Temperature (0.0 - 1.0)

- **Low (0.2-0.4)**: Less exploration, more exploitation
- **Medium (0.6-0.8)**: Balanced exploration/exploitation ✅ Recommended
- **High (0.9-1.0)**: High exploration, more diverse but risky

### Samples per Problem (1-8)

- **Low (1-2)**: Faster but less diverse
- **Medium (4-6)**: Good balance ✅ Recommended
- **High (7-8)**: More compute, better coverage

### Learning Rate (1e-6 - 1e-4)

- **Low (1e-6)**: Very stable but slow
- **Medium (5e-5)**: Good balance ✅ Recommended
- **High (1e-4)**: Faster but risk instability

### PPO Clipping (0.7-1.3)

- **Tight (0.9-1.1)**: Very conservative updates
- **Medium (0.8-1.2)**: Standard PPO ✅ Recommended
- **Loose (0.7-1.3)**: Larger updates, less stable

## Results

### Expected Improvements

On MBPP with 20 iterations:

| Metric | Baseline | After RLVR | Improvement |
|--------|----------|------------|-------------|
| Pass Rate | ~5-10% | ~30-50% | +25-40 pp |
| Mean Reward | ~-0.2 | ~0.4-0.6 | +0.6-0.8 |

### Sample Output

```
================================================================================
FINAL RESULTS
================================================================================
Metric                         Baseline        RLVR            Improvement
--------------------------------------------------------------------------------
Pass Rate                      8.0%            42.0%           +34.0pp
Mean Reward                    -0.150          0.550           +0.700
================================================================================
```

## Code Execution Safety

⚠️ **Important**: This implementation uses `exec()` to run generated code. In production:

1. Use sandboxed execution environments (Docker, gVisor)
2. Implement resource limits (CPU, memory, time)
3. Restrict filesystem and network access
4. Add input validation and sanitization

For research purposes on MBPP, the current implementation is acceptable since:
- MBPP problems are simple and well-vetted
- Timeout limits prevent infinite loops
- Namespace isolation prevents global state pollution

## Troubleshooting

### Problem: Pass rate not improving

**Solutions:**
- Increase temperature for more exploration
- Increase samples per problem
- Start from an SFT checkpoint
- Check if code execution is working correctly

### Problem: Training is unstable

**Solutions:**
- Decrease learning rate
- Tighten PPO clipping thresholds
- Reduce samples per problem
- Add gradient clipping

### Problem: Model generates invalid code

**Solutions:**
- Start from SFT checkpoint (model needs basic syntax knowledge)
- Adjust reward function to penalize syntax errors more
- Increase training iterations

## Advanced Topics

### Curriculum Learning

Start with easier problems and gradually increase difficulty:

```python
# Sort problems by difficulty (e.g., by test count)
sorted_problems = sorted(dataset, key=lambda x: len(x['test_list']))

# Train on easier problems first
for iteration in range(num_iterations):
    difficulty_threshold = min(10, 3 + iteration // 5)
    available_problems = [p for p in sorted_problems 
                         if len(p['test_list']) <= difficulty_threshold]
    # Sample from available_problems...
```

### Reward Shaping

Customize rewards for your use case:

```python
def compute_reward_with_efficiency(all_passed, num_passed, total_tests, code_length):
    base_reward = compute_reward(all_passed, num_passed, total_tests)
    
    # Bonus for concise solutions
    if all_passed and code_length < 100:
        base_reward += 0.2
    
    return base_reward
```

### Multi-Turn Refinement

Allow the model to refine solutions based on test feedback:

```python
# First attempt
initial_solution = sample_solution(problem)
passed, num_passed, error = execute_tests(initial_solution)

# If failed, provide error feedback and retry
if not passed:
    refinement_prompt = f"Previous attempt failed: {error}\nPlease fix the code."
    refined_solution = sample_solution(problem, context=refinement_prompt)
```

## References

- [PPO: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [AlphaCode: Competition-Level Code Generation](https://arxiv.org/abs/2203.07814)
- [CodeRL: Mastering Code Generation through Pretrained Models and Deep RL](https://arxiv.org/abs/2207.01780)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{posttrain-lab-rlvr,
  title={RLVR: Reinforcement Learning with Verifiable Rewards},
  author={PostTrain-Lab Contributors},
  year={2026},
  url={https://github.com/yourusername/PostTrain-Lab}
}
```

## License

MIT License - See LICENSE file for details

