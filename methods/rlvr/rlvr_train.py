"""
RLVR: Reinforcement Learning with Verifiable Rewards

This script demonstrates RLVR training on the MBPP dataset. Unlike SFT which learns 
from correct examples, RLVR allows the model to explore different solutions and learn 
from automated test execution feedback.

Key Concepts:
- Sample multiple solutions from the model
- Execute code against test cases to get verifiable rewards
- Use policy gradient methods (PPO) to reinforce successful solutions
- Learn from both successes and failures
"""

import tinker
from tinker import types
from transformers import AutoTokenizer
from datasets import load_dataset
from tinker_cookbook import renderers
from dotenv import load_dotenv
import numpy as np
import re
import signal
from typing import Dict, List, Tuple
from collections import defaultdict

load_dotenv()


# ============================================================================
# Code Execution and Reward Functions
# ============================================================================

def extract_code_from_response(response_text: str) -> str:
    """
    Extract Python code from model response.
    Handles both markdown code blocks and plain code.
    """
    # Try to extract from markdown code block
    code_block_pattern = r"```python\s*\n(.*?)\n```"
    match = re.search(code_block_pattern, response_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # Try without language specifier
    code_block_pattern = r"```\s*\n(.*?)\n```"
    match = re.search(code_block_pattern, response_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # Return as-is if no code block found
    return response_text.strip()


def execute_code_with_tests(
    code: str, 
    test_cases: List[str], 
    timeout: int = 2
) -> Tuple[bool, int, str]:
    """
    Execute code against test cases in a safe environment.
    
    Returns:
        (all_passed, num_passed, error_message)
    """
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")
    
    num_passed = 0
    error_msg = ""
    
    try:
        # Create isolated namespace
        namespace = {}
        
        # Execute the function definition
        exec(code, namespace)
        
        # Run each test case
        for i, test in enumerate(test_cases):
            try:
                # Set timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                
                # Execute test
                exec(test, namespace)
                num_passed += 1
                
                # Cancel timeout
                signal.alarm(0)
                
            except TimeoutError:
                error_msg = f"Test {i+1} timed out"
                signal.alarm(0)
                break
            except AssertionError as e:
                error_msg = f"Test {i+1} failed: {str(e)}"
                break
            except Exception as e:
                error_msg = f"Test {i+1} error: {type(e).__name__}: {str(e)}"
                break
        
        all_passed = (num_passed == len(test_cases))
        return all_passed, num_passed, error_msg
        
    except SyntaxError as e:
        return False, 0, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, 0, f"Execution error: {type(e).__name__}: {str(e)}"


def compute_reward(all_passed: bool, num_passed: int, total_tests: int) -> float:
    """
    Compute reward based on test results.
    
    Reward structure:
    - Pass all tests: +1.0
    - Partial pass: +0.5 * (num_passed / total_tests)
    - Fail all: -0.5
    """
    if all_passed:
        return 1.0
    elif num_passed > 0:
        return 0.5 * (num_passed / total_tests)
    else:
        return -0.5


# ============================================================================
# Sampling and Rollout Generation
# ============================================================================

def build_prompt_messages(example: Dict) -> List[Dict]:
    """Build prompt messages for code generation."""
    tests = "\n".join(example["test_list"])
    
    return [
        {
            "role": "system",
            "content": "You are a senior Python engineer. Write correct, test-passing code."
        },
        {
            "role": "user",
            "content": f"""Problem:
                {example['text']}

                Tests:
                {tests}

                Write a Python function that solves this problem."""
        }
    ]


def generate_rollouts(
    sampler: tinker.SamplingClient,
    renderer,
    example: Dict,
    num_samples: int = 4,
    temperature: float = 0.8,
    max_tokens: int = 400
) -> List[Dict]:
    """
    Generate multiple solution attempts for a problem.
    
    Returns list of rollouts, each containing:
    - prompt: ModelInput
    - tokens: generated tokens
    - logprobs: log probabilities
    - code: extracted code
    - reward: computed reward
    - passed: whether all tests passed
    """
    messages = build_prompt_messages(example)
    prompt = renderer.build_generation_prompt(messages)
    stop_sequences = renderer.get_stop_sequences()
    
    # Sample multiple solutions
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences
    )
    
    result = sampler.sample(
        prompt=prompt,
        num_samples=num_samples,
        sampling_params=sampling_params
    ).result()
    
    rollouts = []
    
    for seq in result.sequences:
        # Parse response
        response, parse_success = renderer.parse_response(seq.tokens)
        
        if not parse_success:
            # Failed to parse - give negative reward
            rollouts.append({
                "prompt": prompt,
                "tokens": seq.tokens,
                "logprobs": seq.logprobs if seq.logprobs else [0.0] * len(seq.tokens),
                "code": "",
                "reward": -0.5,
                "passed": False,
                "error": "Failed to parse response"
            })
            continue
        
        # Extract code
        code = extract_code_from_response(response["content"])
        
        # Execute and get reward
        all_passed, num_passed, error_msg = execute_code_with_tests(
            code, example["test_list"]
        )
        reward = compute_reward(all_passed, num_passed, len(example["test_list"]))
        
        rollouts.append({
            "prompt": prompt,
            "tokens": seq.tokens,
            "logprobs": seq.logprobs if seq.logprobs else [0.0] * len(seq.tokens),
            "code": code,
            "reward": reward,
            "passed": all_passed,
            "num_passed": num_passed,
            "error": error_msg if not all_passed else ""
        })
    
    return rollouts


# ============================================================================
# Advantage Computation and Data Preparation
# ============================================================================

def compute_advantages(rollouts: List[Dict], normalize: bool = True) -> List[Dict]:
    """
    Compute advantages for each rollout.
    
    Uses group-based advantage normalization:
    advantage = reward - mean(rewards_in_group)
    
    This encourages the model to favor better solutions within each problem.
    """
    if not rollouts:
        return rollouts
    
    rewards = np.array([r["reward"] for r in rollouts])
    
    if normalize:
        # Center advantages around mean
        mean_reward = rewards.mean()
        advantages = rewards - mean_reward
    else:
        advantages = rewards
    
    # Add advantages to rollouts
    for rollout, advantage in zip(rollouts, advantages):
        rollout["advantage"] = float(advantage)
    
    return rollouts


def prepare_training_data(rollouts: List[Dict]) -> List[types.Datum]:
    """
    Convert rollouts into training data for PPO.
    """
    training_data = []
    
    for rollout in rollouts:
        # Combine prompt and completion tokens
        prompt_tokens = rollout["prompt"].to_ints()
        completion_tokens = rollout["tokens"]
        all_tokens = prompt_tokens + completion_tokens
        
        # Logprobs for completion only
        sampling_logprobs = rollout["logprobs"]
        
        # Advantage is constant across all tokens in the sequence
        advantage = rollout["advantage"]
        advantages = [advantage] * len(completion_tokens)
        
        # Create datum
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(all_tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": all_tokens[1:],
                "logprobs": sampling_logprobs,
                "advantages": advantages
            }
        )
        training_data.append(datum)
    
    return training_data


# ============================================================================
# Main Training Loop
# ============================================================================

def train_rlvr(config: Dict):
    """
    Main RLVR training loop.
    """
    print("="*80)
    print("RLVR Training: Reinforcement Learning with Verifiable Rewards")
    print("="*80)
    
    # Load dataset
    print("\n📚 Loading MBPP dataset...")
    dataset = load_dataset("google-research-datasets/mbpp", split="train")
    train_dataset = dataset.select(range(config["num_train_problems"]))
    print(f"   Training on {len(train_dataset)} problems")
    
    # Initialize clients
    print("\n🔧 Initializing Tinker clients...")
    service_client = tinker.ServiceClient()
    
    if config.get("start_from_checkpoint"):
        training_client = service_client.create_training_client_from_state(
            path=config["start_from_checkpoint"]
        )
        print(f"   Loaded from checkpoint: {config['start_from_checkpoint']}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config["base_model"],
            rank=config["lora_rank"]
        )
        print(f"   Created new training client: {config['base_model']}")
    
    # Setup tokenizer and renderer
    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer("llama3", tokenizer)
    
    # Training metrics
    metrics_history = {
        "iteration": [],
        "mean_reward": [],
        "pass_rate": [],
        "mean_advantage": [],
        "loss": []
    }
    
    # Main training loop
    print(f"\n🚀 Starting training for {config['num_iterations']} iterations...")
    print("="*80)
    
    for iteration in range(config["num_iterations"]):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{config['num_iterations']}")
        print(f"{'='*80}")
        
        # Create sampler from current weights
        sampler = training_client.save_weights_and_get_sampling_client(
            f"rlvr-iter-{iteration}"
        )
        
        # Sample problems for this iteration
        problem_indices = np.random.choice(
            len(train_dataset),
            size=config["problems_per_iteration"],
            replace=False
        )
        
        # Generate rollouts for all problems
        all_rollouts = []
        iteration_stats = defaultdict(list)
        
        print(f"\n🎲 Generating rollouts for {config['problems_per_iteration']} problems...")
        
        for prob_idx in problem_indices:
            example = train_dataset[int(prob_idx)]
            
            # Generate multiple solutions
            rollouts = generate_rollouts(
                sampler,
                renderer,
                example,
                num_samples=config["samples_per_problem"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )
            
            # Compute advantages per problem (group normalization)
            rollouts = compute_advantages(rollouts, normalize=True)
            all_rollouts.extend(rollouts)
            
            # Track stats
            for rollout in rollouts:
                iteration_stats["rewards"].append(rollout["reward"])
                iteration_stats["passed"].append(rollout["passed"])
                iteration_stats["advantages"].append(rollout["advantage"])
        
        # Prepare training data
        training_data = prepare_training_data(all_rollouts)
        
        # Compute statistics
        mean_reward = np.mean(iteration_stats["rewards"])
        pass_rate = np.mean(iteration_stats["passed"])
        mean_advantage = np.mean(np.abs(iteration_stats["advantages"]))
        
        print(f"\n📊 Rollout Statistics:")
        print(f"   Total rollouts: {len(all_rollouts)}")
        print(f"   Mean reward: {mean_reward:.3f}")
        print(f"   Pass rate: {pass_rate:.1%}")
        print(f"   Mean |advantage|: {mean_advantage:.3f}")
        
        # Perform PPO update(s)
        print(f"\n⚡ Performing {config['ppo_epochs']} PPO update(s)...")
        
        for epoch in range(config["ppo_epochs"]):
            # Forward-backward pass with PPO loss
            fwd_bwd_future = training_client.forward_backward(
                training_data,
                loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": config["ppo_clip_low"],
                    "clip_high_threshold": config["ppo_clip_high"]
                }
            )
            
            # Optimizer step
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=config["learning_rate"])
            )
            
            # Wait for results
            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()
            
            # Compute loss for logging
            if "loss:sum" in fwd_bwd_result.metrics:
                loss = fwd_bwd_result.metrics["loss:sum"]
            else:
                loss = 0.0
        
        print(f"   PPO loss: {loss:.4f}")
        
        # Save metrics
        metrics_history["iteration"].append(iteration + 1)
        metrics_history["mean_reward"].append(mean_reward)
        metrics_history["pass_rate"].append(pass_rate)
        metrics_history["mean_advantage"].append(mean_advantage)
        metrics_history["loss"].append(loss)
        
        # Print progress summary
        if (iteration + 1) % 5 == 0:
            print(f"\n{'='*80}")
            print(f"📈 Progress Summary (Iteration {iteration + 1})")
            print(f"{'='*80}")
            recent_rewards = metrics_history["mean_reward"][-5:]
            recent_pass_rates = metrics_history["pass_rate"][-5:]
            print(f"   Recent mean reward: {np.mean(recent_rewards):.3f}")
            print(f"   Recent pass rate: {np.mean(recent_pass_rates):.1%}")
            print(f"{'='*80}")
    
    print("\n✅ Training complete!")
    
    # Save final checkpoint
    print("\n💾 Saving final checkpoint...")
    final_checkpoint = training_client.save_state("rlvr-final").result()
    print(f"   Checkpoint saved: {final_checkpoint.path}")
    
    # Create final sampler
    final_sampler = training_client.save_weights_and_get_sampling_client("rlvr-final-sampler")
    print("   Final sampler created!")
    
    return training_client, final_sampler, metrics_history


def evaluate_model(
    sampler: tinker.SamplingClient,
    renderer,
    dataset,
    test_indices: List[int],
    name: str = "Model"
) -> Dict:
    """
    Evaluate a model on test problems.
    """
    results = []
    
    for idx in test_indices:
        problem = dataset[int(idx)]
        rollouts = generate_rollouts(
            sampler,
            renderer,
            problem,
            num_samples=1,
            temperature=0.2
        )
        results.append(rollouts[0])
    
    pass_rate = np.mean([r['passed'] for r in results])
    mean_reward = np.mean([r['reward'] for r in results])
    
    return {
        "name": name,
        "results": results,
        "pass_rate": pass_rate,
        "mean_reward": mean_reward
    }


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Training configuration
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
        # "start_from_checkpoint": "tinker://your-sft-checkpoint",  # Optional
    }
    
    print("\n🧠 RLVR Training Configuration:")
    print("="*80)
    for key, value in config.items():
        if key != "start_from_checkpoint" or value:
            print(f"  {key:.<40} {value}")
    print("="*80)
    
    # Train model
    training_client, final_sampler, metrics = train_rlvr(config)
    
    # Evaluation
    print("\n" + "="*80)
    print("📊 EVALUATION")
    print("="*80)
    
    # Load dataset and setup
    dataset = load_dataset("google-research-datasets/mbpp", split="train")
    service_client = tinker.ServiceClient()
    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer("llama3", tokenizer)
    
    # Create baseline sampler
    print("\n🔵 Creating baseline sampler...")
    baseline_sampler = service_client.create_sampling_client(
        base_model=config["base_model"]
    )
    
    # Test on held-out problems
    test_indices = list(range(200, 210))
    print(f"\n🧪 Testing on {len(test_indices)} held-out problems...")
    
    print("\n   Evaluating baseline...")
    baseline_eval = evaluate_model(
        baseline_sampler, renderer, dataset, test_indices, "Baseline"
    )
    
    print("   Evaluating RLVR model...")
    rlvr_eval = evaluate_model(
        final_sampler, renderer, dataset, test_indices, "RLVR"
    )
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"{'Metric':<30} {'Baseline':<15} {'RLVR':<15} {'Improvement'}")
    print("-"*80)
    print(f"{'Pass Rate':<30} {baseline_eval['pass_rate']:<15.1%} "
          f"{rlvr_eval['pass_rate']:<15.1%} "
          f"{(rlvr_eval['pass_rate'] - baseline_eval['pass_rate'])*100:+.1f}pp")
    print(f"{'Mean Reward':<30} {baseline_eval['mean_reward']:<15.3f} "
          f"{rlvr_eval['mean_reward']:<15.3f} "
          f"{rlvr_eval['mean_reward'] - baseline_eval['mean_reward']:+.3f}")
    print("="*80)
    
    # Show training progress
    print("\n📈 Training Progress:")
    print(f"   Initial mean reward: {metrics['mean_reward'][0]:.3f}")
    print(f"   Final mean reward: {metrics['mean_reward'][-1]:.3f}")
    print(f"   Improvement: {metrics['mean_reward'][-1] - metrics['mean_reward'][0]:.3f}")
    print()
    print(f"   Initial pass rate: {metrics['pass_rate'][0]:.1%}")
    print(f"   Final pass rate: {metrics['pass_rate'][-1]:.1%}")
    print(f"   Improvement: {(metrics['pass_rate'][-1] - metrics['pass_rate'][0]) * 100:.1f} pp")
    
    print("\n✨ RLVR training complete! ✨\n")

