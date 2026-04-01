# RL Training with Tinker and Prime Intellect Environments Hub

## Overview

This recipe enables Reinforcement Learning (RL) training for Large Language Models (LLMs) using environments from the [Verifiers](https://github.com/primeintellect-ai/verifiers) library and [Prime Intellect's Environments Hub](https://app.primeintellect.ai/dashboard/environments). The integration allows any text-based environment from the Hub to be seamlessly used with Tinker's RL training infrastructure.

The recipe bridges two powerful frameworks:
- **Tinker**: A post-training framework for LLMs with distributed training capabilities
- **Verifiers**: A library for creating RL environments with standardized interfaces for LLM evaluation

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Verifiers Environment                     │
│  (from Prime Intellect Environments Hub)                     │
│  Examples: reverse-text, alphabet-sort, math-python, wordle  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ RolloutInput/State
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              VerifiersEnvGroupBuilder                        │
│  • Converts environment inputs to Tinker format              │
│  • Manages rollout configuration                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ TrajectoryGroup
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           TinkerAsyncOpenAIClient                            │
│  • OpenAI-compatible interface backed by Tinker              │
│  • Handles model sampling and token generation               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Completions
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Tinker RL Training Loop                     │
│  • PPO-style optimization with LoRA adapters                 │
│  • Batch rollout and gradient updates                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Files

1. **`train.py`**: Main training script
   - Defines CLI configuration
   - Integrates with Tinker's RL training loop
   - Implements custom rollout logic for Verifiers environments

2. **`evaluate.py`**: Offline evaluation script
   - Evaluates trained models or base models
   - Generates multiple rollouts per example
   - Computes reward statistics and metrics

3. **`verifiers_env.py`**: Environment wrapper
   - `VerifiersRLDatasetBuilder`: Loads datasets from environments
   - `VerifiersEnvGroupBuilder`: Prepares rollout inputs for environment groups
   - `convert_states_to_trajectory_group()`: Converts Verifiers States to Tinker TrajectoryGroups

4. **`tinker_openai.py`**: OpenAI-compatible client
   - `TinkerAsyncOpenAIClient`: Drop-in replacement for OpenAI's AsyncOpenAI
   - Implements `chat.completions.create()` and `completions.create()`
   - Uses Tinker's SamplingClient as the backend

## Installation and Setup

### Prerequisites

- Tinker installed and configured
- `verifiers` library installed
- `prime` CLI tool for installing environments

### Installing Environments

Install environments from the Environments Hub using the `prime` CLI:

```bash
# Install prime CLI
uv tool install prime  # or pipx install prime

# Install an environment
prime env install user/env-id

# Examples
prime env install primeintellect/reverse-text
prime env install primeintellect/alphabet-sort
prime env install primeintellect/math-python
prime env install will/wordle
```

Each environment is a self-contained Python package that gets installed into your environment.

## Usage

### Training

Basic training command:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.train \
    vf_env_id=env-id \
    vf_env_args='{}' \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    learning_rate=1e-5 \
    groups_per_batch=32 \
    group_size=8
```

#### Training Configuration Parameters

**Model Configuration:**
- `model_name` (str): Base model to train (default: `"Qwen/Qwen3-4B-Instruct-2507"`)
- `lora_rank` (int): LoRA rank for efficient fine-tuning (default: 32)

**Environment Configuration:**
- `vf_env_id` (str): Environment ID (just the `env-id` part, not `user/env-id`) (default: `"reverse-text"`)
- `vf_env_args` (str | None): JSON string of arguments to pass to the environment (default: `None`)
- `dataset_n` (int): Number of examples from dataset to use, -1 for all (default: -1)
- `dataset_seed` (int | None): Random seed for dataset sampling (default: `None`)

**Training Hyperparameters:**
- `group_size` (int): Number of rollouts per environment group (default: 8)
- `groups_per_batch` (int): Number of groups per training batch (default: 32)
- `num_substeps` (int): Number of gradient updates per batch (default: 1)
- `learning_rate` (float): Learning rate for optimizer (default: 1e-5)
- `max_tokens` (int): Maximum tokens to generate per completion (default: 512)
- `temperature` (float): Sampling temperature (default: 1.0)
- `kl_penalty_coef` (float): KL divergence penalty coefficient (default: 0.0)
- `max_concurrent_generation` (int): Max concurrent generation requests, -1 for unlimited (default: -1)
- `max_concurrent_scoring` (int): Max concurrent scoring requests, -1 for unlimited (default: -1)

**Logging Configuration:**
- `eval_every` (int): Steps between evaluations, 0 to disable (default: 0)
- `save_every` (int): Steps between checkpoint saves (default: 10)
- `log_path` (str | None): Path to save logs and checkpoints (default: auto-generated in `/tmp`)
- `wandb_project` (str | None): Weights & Biases project name (default: `None`)
- `wandb_name` (str | None): Run name for W&B (default: auto-generated)
- `behavior_if_log_dir_exists` (str): Behavior when log directory exists: "ask", "overwrite", "fail", or "use" (default: "ask")

#### Example Training Commands

**Basic reverse-text training:**
```bash
python -m tinker_cookbook.recipes.verifiers_rl.train \
    vf_env_id=reverse-text
```

**Math problem solving with custom parameters:**
```bash
python -m tinker_cookbook.recipes.verifiers_rl.train \
    vf_env_id=math-python \
    model_name="Qwen/Qwen3-8B-Instruct-2507" \
    learning_rate=5e-6 \
    lora_rank=64 \
    groups_per_batch=64 \
    group_size=16 \
    max_tokens=1024 \
    temperature=0.8
```

**Training with W&B logging:**
```bash
python -m tinker_cookbook.recipes.verifiers_rl.train \
    vf_env_id=alphabet-sort \
    wandb_project=my-rl-experiments \
    wandb_name=alphabet-sort-run-1 \
    log_path=./logs/alphabet-sort
```

**Environment with custom arguments:**
```bash
python -m tinker_cookbook.recipes.verifiers_rl.train \
    vf_env_id=custom-env \
    vf_env_args='{"difficulty": "hard", "max_steps": 5}'
```

### Evaluation

Evaluate a trained model or base model offline:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.evaluate \
    vf_env_id=env-id \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    model_path=/path/to/checkpoint \
    num_examples=10 \
    rollouts_per_example=5
```

#### Evaluation Configuration Parameters

- `model_name` (str | None): Base model name (auto-detected from checkpoint if not provided)
- `model_path` (str | None): Path to trained checkpoint (from `checkpoints.jsonl` `sampler_path`)
- `vf_env_id` (str): Environment ID (default: `"reverse-text"`)
- `vf_env_args` (str | None): JSON string of environment arguments
- `num_examples` (int): Number of dataset examples to evaluate (default: 5)
- `rollouts_per_example` (int): Number of rollouts per example (default: 3)
- `max_concurrent` (int): Maximum concurrent requests (default: 32)
- `max_tokens` (int): Maximum tokens per completion (default: 1024)
- `temperature` (float): Sampling temperature (default: 1.0)

#### Example Evaluation Commands

**Evaluate base model:**
```bash
python -m tinker_cookbook.recipes.verifiers_rl.evaluate \
    vf_env_id=reverse-text \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    num_examples=20 \
    rollouts_per_example=10
```

**Evaluate trained checkpoint:**
```bash
python -m tinker_cookbook.recipes.verifiers_rl.evaluate \
    vf_env_id=reverse-text \
    model_path=/tmp/tinker-examples/verifiers_rl/run_123/checkpoints/step_32/sampler \
    num_examples=50 \
    rollouts_per_example=5
```

## How It Works

### Training Flow

1. **Dataset Loading**:
   - `VerifiersRLDatasetBuilder` loads the environment using `vf.load_environment()`
   - Retrieves dataset from environment with `env.get_dataset()`
   - Each dataset row contains: prompt, example_id, task, answer (optional), info (optional)

2. **Rollout Generation**:
   - For each training step, the custom `custom_do_group_rollout()` function is called
   - `VerifiersEnvGroupBuilder` prepares `group_size` rollout inputs from a dataset example
   - `TinkerAsyncOpenAIClient` is created with the current policy (model + LoRA adapter)
   - Environment's `run_group()` method executes rollouts with the Tinker-backed client
   - Returns `vf.State` objects containing trajectories and rewards

3. **State Conversion**:
   - `convert_states_to_trajectory_group()` converts Verifiers States to Tinker TrajectoryGroups
   - Extracts prompt IDs, completion IDs, and logprobs from each trajectory step
   - Creates `Transition` objects with observations, actions, and rewards
   - Final reward is assigned to the entire trajectory

4. **Policy Update**:
   - Tinker's RL training loop performs PPO-style optimization
   - Updates LoRA adapter weights based on trajectory rewards
   - Applies KL penalty to prevent drift from base model (if configured)

5. **Checkpointing and Logging**:
   - Saves checkpoints at regular intervals (controlled by `save_every`)
   - Logs metrics to console and optionally to Weights & Biases
   - Stores training state for resumption

### Evaluation Flow

1. **Model Loading**:
   - Loads base model or checkpoint using Tinker's ServiceClient
   - If checkpoint provided, automatically detects base model

2. **Environment Setup**:
   - Loads environment and initializes TinkerAsyncOpenAIClient

3. **Rollout Execution**:
   - Runs multiple rollouts per example using `env.evaluate_sync()`
   - Collects rewards and custom metrics

4. **Result Aggregation**:
   - Computes statistics (mean, std) across all rollouts
   - Displays sample prompts, completions, and rewards
   - Shows per-rollout performance breakdown

## TinkerAsyncOpenAIClient

A standalone component that can be adapted for other applications requiring OpenAI API compatibility with Tinker backends.

### Features

- **Full OpenAI API Compatibility**: Drop-in replacement for `AsyncOpenAI`
- **Chat Completions**: Implements `chat.completions.create()` with message-based interface
- **Text Completions**: Implements `completions.create()` for raw text prompts
- **Logprobs Support**: Returns token-level log probabilities
- **Renderer Integration**: Uses Tinker's renderer system for proper prompt formatting and response parsing

### Usage Example

```python
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient

# Create Tinker sampling client
service = tinker.ServiceClient()
sampling = service.create_sampling_client(base_model="Qwen/Qwen3-4B-Instruct-2507")

# Initialize tokenizer and renderer
tokenizer = get_tokenizer("Qwen/Qwen3-4B-Instruct-2507")
renderer = renderers.get_renderer("qwen", tokenizer)

# Create OpenAI-compatible client
client = TinkerAsyncOpenAIClient(sampling, renderer, tokenizer)

# Use like standard OpenAI client
response = await client.chat.completions.create(
    model="tinker",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Implementation Details

**Supported Parameters:**
- `messages`: List of message dictionaries with `role` and `content`
- `max_tokens` or `max_completion_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (0.0 to 2.0)
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-K sampling parameter
- `stop`: Stop sequences (defaults to renderer's stop sequences)

**Limitations:**
- Streaming (`stream=True`) is not supported
- Tool calling is not yet supported

**Custom Attributes:**
The client adds non-standard attributes to responses for debugging:
- `response.prompt_token_ids`: Token IDs of the prompt
- `response.choices[0].token_ids`: Token IDs of the completion

## Understanding Verifiers Environments

### Environment Interface

Verifiers environments implement a standardized interface:

```python
class Environment:
    def get_dataset(self, n: int = -1, seed: int | None = None) -> Dataset:
        """Get dataset of prompts/tasks."""
        
    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client: AsyncOpenAI,
        model: str,
        gen_sampling_args: dict,
        gen_sem: Semaphore | None = None,
        score_sem: Semaphore | None = None
    ) -> list[State]:
        """Run a group of rollouts and return states with rewards."""
        
    def evaluate_sync(
        self,
        client: AsyncOpenAI,
        model: str,
        num_examples: int,
        rollouts_per_example: int,
        max_concurrent: int,
        sampling_args: dict
    ) -> GenerateOutputs:
        """Evaluate model on multiple examples."""
```

### RolloutInput

Input to a single environment rollout:

```python
@dataclass
class RolloutInput:
    prompt: Messages  # Initial prompt messages
    answer: str       # Ground truth answer (if available)
    task: str         # Task description
    info: dict        # Additional metadata
    example_id: int   # Dataset example identifier
```

### State

Output from a single environment rollout:

```python
@dataclass
class State:
    reward: float                    # Final reward for the trajectory
    trajectory: list[TrajectoryStep] # Sequence of interaction steps
    metrics: dict[str, float | int]  # Custom environment metrics
    # ... other fields
```

Each `TrajectoryStep` contains:
- `tokens`: Dict with `prompt_ids`, `completion_ids`, `completion_logprobs`
- Other step-specific data from the environment

### Creating Custom Environments

To create your own environment for the Hub:

1. Implement the `Environment` interface
2. Define your reward function
3. Specify dataset generation or loading
4. Package as a self-contained Python module
5. Upload to Environments Hub via Prime Intellect platform

See [Verifiers documentation](https://github.com/primeintellect-ai/verifiers) for detailed guide.

## Training Deep Dive

### PPO-Style Optimization

The recipe uses Proximal Policy Optimization (PPO) concepts:

1. **Rollout Collection**: Generate multiple trajectories using current policy
2. **Advantage Estimation**: Compute advantages based on trajectory rewards
3. **Policy Update**: Update LoRA adapter to maximize expected reward
4. **KL Constraint**: Optionally constrain updates to prevent large deviations

### Batch Configuration

The training uses a two-level batching strategy:

- **Group Size** (`group_size`): Number of rollouts per environment prompt
  - Higher values → More samples per prompt, better gradient estimates
  - Lower values → Faster iteration, more environment diversity
  
- **Groups Per Batch** (`groups_per_batch`): Number of environment prompts per batch
  - Higher values → More stable gradients, higher memory usage
  - Lower values → Faster updates, more frequent checkpoints

**Total Rollouts per Batch** = `group_size × groups_per_batch`

For example, with `group_size=8` and `groups_per_batch=32`:
- Each batch processes 32 different environment prompts
- Each prompt generates 8 rollout trajectories
- Total of 256 trajectories per batch

### LoRA Adapters

The recipe uses LoRA (Low-Rank Adaptation) for efficient training:

- **Benefits**:
  - Much faster training than full fine-tuning
  - Lower memory footprint
  - Preserves base model capabilities
  - Easy to merge back into base model

- **LoRA Rank** (`lora_rank`):
  - Common values: 8, 16, 32, 64, 128
  - Higher rank → More capacity, slower training, higher memory
  - Lower rank → Faster training, less capacity
  - Recommended: Start with 32, increase if performance plateaus

### Concurrent Processing

The recipe supports concurrent generation and scoring for throughput:

- `max_concurrent_generation`: Limits parallel model sampling requests
- `max_concurrent_scoring`: Limits parallel reward computation requests

Setting to -1 (unlimited) maximizes throughput but may overwhelm resources. Adjust based on your hardware.

### KL Penalty

The `kl_penalty_coef` parameter controls how much the policy is penalized for diverging from the base model:

- **0.0** (default): No penalty, policy can freely explore
- **0.01-0.1**: Light penalty, encourages staying close to base model
- **> 0.1**: Strong penalty, minimal deviation from base model

Higher KL penalty helps:
- Prevent mode collapse
- Preserve general language capabilities
- Stabilize training

But may:
- Limit task-specific performance gains
- Slow convergence

## Expected Performance

### Reverse-Text Example

The reverse-text task (reversing a string) typically shows:

- **Initial Performance**: ~0.2 reward (random baseline)
- **After 32 Steps**: ~0.35 reward
- **Training Time**: Varies based on hardware and batch configuration

This demonstrates that the model learns to improve at the task through RL.

### Monitoring Training

**Key Metrics to Watch:**

1. **Average Reward**: Should generally increase over time
2. **Reward Standard Deviation**: High variance indicates exploration
3. **KL Divergence**: Monitor drift from base model
4. **Loss**: Should decrease, though not perfectly monotonic in RL
5. **Environment-Specific Metrics**: Depend on the environment (accuracy, completion rate, etc.)

**Checkpoints:**

Training saves checkpoints at intervals specified by `save_every`. Each checkpoint contains:
- LoRA adapter weights
- Training state (optimizer, scheduler)
- Metadata (step number, config)

Location: `{log_path}/checkpoints/step_{N}/sampler`

**Logs:**

- Console output shows per-step metrics
- `checkpoints.jsonl`: Checkpoint metadata
- W&B dashboard (if configured): Real-time metrics and plots

## Troubleshooting

### Common Issues

**1. Environment Not Found**

```
Error: Environment 'env-id' not found
```

**Solution**: Install the environment first:
```bash
prime env install user/env-id
```

**2. Qwen3 Models Stripping `<think>` Tags**

Some environments use custom `<think>` sections for reasoning. Qwen3 tokenizers strip these automatically, which may affect reward calculations.

**Solutions**:
- Modify the environment's reward function to handle stripped tags
- Use a different tokenizer chat template
- Modify the renderer to preserve `<think>` tags
- Use a different base model family

**3. Out of Memory (OOM)**

**Symptoms**: Training crashes with CUDA OOM errors

**Solutions**:
- Reduce `groups_per_batch` or `group_size`
- Reduce `lora_rank`
- Reduce `max_tokens`
- Use a smaller base model
- Enable gradient checkpointing (if supported)

**4. Slow Training**

**Solutions**:
- Increase `max_concurrent_generation` and `max_concurrent_scoring`
- Use smaller `max_tokens` value
- Reduce `group_size` if you have many cores
- Consider using a smaller model for faster iteration

**5. Reward Not Improving**

**Potential Causes**:
- Learning rate too high or too low
- Task too difficult for model capacity
- Insufficient exploration (temperature too low)
- KL penalty too high

**Solutions**:
- Adjust `learning_rate` (try 5e-6 to 5e-5)
- Increase `temperature` for more exploration
- Reduce `kl_penalty_coef`
- Verify environment rewards are correct with base model evaluation

## Advanced Topics

### Custom Rollout Logic

The recipe overrides Tinker's default rollout function with custom logic:

```python
async def custom_do_group_rollout(
    builder: EnvGroupBuilder, policy: TokenCompleter
) -> TrajectoryGroup:
    # Initialize shared resources (tokenizer, renderer, client)
    # Get rollout inputs from builder
    # Run environment group with TinkerAsyncOpenAIClient
    # Convert states to TrajectoryGroup
    return convert_states_to_trajectory_group(states)

train.do_group_rollout = custom_do_group_rollout
```

This allows the Verifiers environment to control the interaction loop while Tinker manages the training loop.

### State-to-Trajectory Conversion

The conversion process:

1. Extract trajectory steps from each `vf.State`
2. For each step, extract:
   - `prompt_ids`: Tokens for the prompt/observation
   - `completion_ids`: Tokens generated by the model
   - `completion_logprobs`: Log probabilities of generated tokens
3. Create `Transition` objects with:
   - Observation (`ob`): ModelInput from prompt IDs
   - Action (`ac`): TokensWithLogprobs from completion
   - Reward: 0.0 for intermediate steps, final reward from State
   - Episode done flag: True only for last step
4. Bundle into `TrajectoryGroup` with final rewards and metrics

This ensures Tinker's RL algorithms can properly compute advantages and policy gradients.

### Extending to New Use Cases

The `TinkerAsyncOpenAIClient` can be used independently for:

- **Testing**: Use Tinker models with OpenAI-compatible test suites
- **Integration**: Connect Tinker to existing applications expecting OpenAI API
- **Development**: Develop against OpenAI API locally with Tinker models
- **Benchmarking**: Evaluate Tinker models on OpenAI-based benchmarks

Simply create the client with your Tinker sampling client and use it anywhere `AsyncOpenAI` is accepted.

## Performance Tuning

### Hardware Considerations

**GPU Memory:**
- Larger models require more VRAM
- LoRA reduces memory compared to full fine-tuning
- Batch size directly impacts memory usage

**CPU/Concurrency:**
- Increase concurrent requests if you have unused CPU
- Environment scoring often happens on CPU
- Balance generation and scoring concurrency

### Hyperparameter Tuning

**Learning Rate:**
- Start with 1e-5
- If loss/reward is noisy: decrease to 5e-6 or 1e-6
- If convergence is slow: increase to 5e-5
- Monitor for instability (exploding rewards/loss)

**Temperature:**
- Higher (1.0-1.5): More exploration, diverse outputs
- Lower (0.5-0.8): More exploitation, consistent outputs
- Very low (<0.5): May reduce learning signal

**Batch Configuration:**
- Larger batches: More stable gradients, slower iteration
- Smaller batches: Faster feedback, noisier gradients
- Rule of thumb: Total rollouts per batch should be 128-512

## Environment Examples

### Reverse Text

**Task**: Reverse a given string

**Reward**: Proportion of correctly reversed characters

**Difficulty**: Easy (baseline ~0.2, trained ~0.35+)

**Use Case**: Simple sanity check, testing setup

### Alphabet Sort

**Task**: Sort words alphabetically

**Reward**: Correctness of sorting

**Difficulty**: Medium

**Use Case**: Multi-step reasoning, sequence manipulation

### Math Python

**Task**: Solve math problems using Python code generation

**Reward**: Correctness of solution execution

**Difficulty**: Hard

**Use Case**: Code generation, mathematical reasoning

### Wordle

**Task**: Play Wordle game and guess the word

**Reward**: Success within guess limit

**Difficulty**: Medium-Hard

**Use Case**: Interactive reasoning, constraint satisfaction

## Best Practices

1. **Start Simple**: Begin with easy environments (reverse-text) to validate setup
2. **Monitor Early**: Check first few steps to ensure training is stable
3. **Baseline First**: Run evaluation on base model to understand initial performance
4. **Checkpoint Often**: Set `save_every` to reasonable intervals (10-20 steps)
5. **Experiment Tracking**: Use W&B for comparing different hyperparameters
6. **Resource Management**: Set appropriate concurrency limits for your hardware
7. **Incremental Scaling**: Start with small batches, scale up once stable

## References

- [Verifiers GitHub](https://github.com/primeintellect-ai/verifiers)
- [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments)
- [Tinker Documentation](https://github.com/primeintellect-ai/tinker)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## Contributing

To add new environments to the Hub:

1. Create an environment following the Verifiers interface
2. Test locally with this recipe
3. Submit to Prime Intellect's Environments Hub
4. Share with the community

For issues or questions:
- Verifiers: [GitHub Issues](https://github.com/primeintellect-ai/verifiers/issues)
- Tinker: [GitHub Issues](https://github.com/primeintellect-ai/tinker/issues)
- Environments Hub: [Prime Intellect Support](https://app.primeintellect.ai)
