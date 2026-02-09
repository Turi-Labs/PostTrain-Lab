# Tinker Cookbook RL Recipes - Explained in Plain English

> **For Non-Programmers** - This document explains what's actually happening in each RL recipe, without assuming you know how to code.

---

## 📑 Table of Contents

1. [RL Basic - Your First RL Run](#1-rl-basic---your-first-rl-run)
2. [RL Loop - Understanding the Training Loop](#2-rl-loop---understanding-the-training-loop)
3. [Math RL - Training on Math Problems](#3-math-rl---training-on-math-problems)
4. [Code RL - Training for Code Generation](#4-code-rl---training-for-code-generation)
5. [Multiplayer RL - Multi-Agent Training](#5-multiplayer-rl---multi-agent-training)

---

## 1. RL Basic - Your First RL Run

### 📝 What This Recipe Does

**Goal:** Train a language model to solve grade-school math problems (GSM8K dataset)

**The Task:** Problems like "If John has 5 apples and buys 3 more, how many does he have?"

**Expected Answer Format:** The model must write its answer inside `\boxed{8}`

### 🎯 The Setup

**Model Used:** Llama-3.1-8B (a language model with 8 billion parameters)

**Dataset:** GSM8K - A collection of 7,473 grade-school math word problems

**Training Configuration:**
- **Batch size: 128** - Process 128 problems at a time
- **Group size: 16** - Generate 16 different answers per problem
- **Learning rate: 4e-5** - How fast the model learns (0.00004)
- **Max tokens: 256** - Maximum length of each answer

### 🔄 What Happens Step by Step

#### Step 1: Present the Problem
```
The model sees:
"A shop has 15 books. They sell 7 books and then receive 
a shipment of 10 more books. How many books do they have now?"
```

#### Step 2: Model Generates 16 Different Answers

The model tries 16 different ways to solve it:

```
Answer 1: "15 - 7 = 8, then 8 + 10 = 18. \boxed{18}" ✅
Answer 2: "15 + 10 = 25, then 25 - 7 = 18. \boxed{18}" ✅
Answer 3: "15 - 7 + 10 = 18. \boxed{18}" ✅
Answer 4: "Let me think... I get \boxed{17}" ❌
Answer 5: "15 - 7 = 8. \boxed{8}" ❌
... (11 more attempts)
```

#### Step 3: Score Each Answer

The reward function checks:
- **Is it formatted correctly?** (Has `\boxed{answer}`)
- **Is the answer correct?** (Uses symbolic math to check)

```
Reward formula:
- Correct answer: +1.0
- Wrong answer but good format: -0.9 (penalty 0.1 for format)
- No boxed format: -1.0
```

In our example:
- Answer 1, 2, 3: **+1.0** (correct!)
- Answer 4: **+0.0** (wrong but formatted)
- Answer 5: **+0.0** (wrong and badly formatted)
- Most others: **0.0**

#### Step 4: Compute Advantages

```
Mean reward = (1.0 + 1.0 + 1.0 + 0.0 + 0.0 + ...) / 16 = 0.25

Advantages:
- Answer 1: 1.0 - 0.25 = +0.75 (MUCH better than average!)
- Answer 2: 1.0 - 0.25 = +0.75 
- Answer 3: 1.0 - 0.25 = +0.75
- Answer 4: 0.0 - 0.25 = -0.25 (worse than average)
- Answer 5: 0.0 - 0.25 = -0.25
```

#### Step 5: Update the Model

The model learns:
- **"Increase probability"** of generating solutions like Answer 1, 2, 3
- **"Decrease probability"** of generating solutions like Answer 4, 5

#### Step 6: Repeat!

Do this for all 128 problems in the batch, then move to the next batch.

### 📈 Expected Results

After 15 iterations (about 15 minutes):
- **Accuracy:** Climbs to ~63%
- **Format rate:** Nearly 100% (model learns to always use `\boxed{}`)
- **Reasoning:** Model develops strategies for solving word problems

### 💡 Key Insight

The model isn't memorizing answers - it's learning **strategies**:
- "Break the problem into steps"
- "Keep track of additions and subtractions"
- "Format the final answer correctly"

---

## 2. RL Loop - Understanding the Training Loop

### 📝 What This Recipe Does

**Goal:** Show the **minimal** code needed for RL training, without extra features

**Same task as RL Basic:** Math problems (GSM8K)

**Why this exists:** So you can understand exactly what's happening without distractions

### 🔍 The Core Loop (Step by Step)

Think of it like teaching someone to solve puzzles:

#### Iteration 1: The First Try

**Morning:**
1. Give student (model) 128 different puzzles
2. For each puzzle, ask student to try 16 different solutions
3. Check which solutions work
4. Tell student which approaches were good/bad
5. Student practices (model updates)

**What the student learns:**
- "Oh, I should try this approach more"
- "That approach didn't work, avoid it"

#### Iteration 2-57: Keep Practicing

Repeat the same process with new batches of puzzles.

**What happens over time:**
- Student gets better at recognizing patterns
- Student develops go-to strategies
- Success rate steadily improves

### 🎓 What Makes This "Minimal"?

Unlike `rl_basic.py`, this version:
- ❌ No evaluation on test set
- ❌ No fancy logging
- ❌ No periodic checkpoints
- ✅ Just the core training loop

**It's like:** The difference between a full cooking recipe (with garnish, presentation tips, variations) vs. just the essential steps (mix ingredients, cook, done).

### 📊 How to See Results

After running, you can plot the learning curve:

```
Y-axis: Average reward
X-axis: Training step

You'll see a curve going up:
  Reward
    ^
1.0 |                    ___---
    |              ___---
0.5 |        ___---
    |  __---
0.0 |--
    +-------------------------> Steps
```

This shows the model is learning!

### 💡 Key Insight

RL training is fundamentally simple:
1. Try stuff
2. See what works
3. Do more of what works
4. Repeat

Everything else is just optimizations and book-keeping!

---

## 3. Math RL - Training on Math Problems

### 📝 What This Recipe Does

**Goal:** Train models on various math problem datasets

**Supports 4 Datasets:**

1. **GSM8K** - Grade school math (elementary level)
   - Example: "Sarah has 12 cookies and gives away 4..."

2. **MATH** - Competition math (high school level) 
   - Example: "Find the derivative of x³ + 2x²..."

3. **Polaris** - 53,000 diverse math problems
   - Mix of easy to hard problems

4. **DeepMath** - 103,000 advanced math problems
   - University-level mathematics

### 🎯 How It Works

#### The Environment Setup

For each math problem, the recipe creates an "environment":

**Think of it like a classroom:**
- **Teacher (Environment):** Poses the question
- **Student (Model):** Tries to solve it
- **Grading System (Reward Function):** Checks the answer

#### Example Walkthrough - GSM8K Problem

**Teacher presents:**
```
"A bakery makes 120 cookies. They sell 35 in the morning
and 48 in the afternoon. How many cookies are left?"
```

**Student's thought process (visible in the response):**
```
"Let me calculate step by step:
- Start with: 120 cookies
- Morning sales: 120 - 35 = 85 remaining
- Afternoon sales: 85 - 48 = 37 remaining
Therefore, \boxed{37} cookies are left."
```

**Grading System checks:**
1. **Format check:** Is the answer in `\boxed{37}`? ✅
2. **Correctness check:** Is 37 the right answer? ✅
3. **Reward:** +1.0 (perfect!)

#### The "Few-Shot" Teaching Trick

Before each problem, the model sees an example:

```
Example given to model:
"How many r's are in strawberry?"

Good answer to learn from:
"Let's spell it: 1)s 2)t 3)r 4)a 5)w 6)b 7)e 8)r 9)r 10)y
The r's are at positions 3, 8, and 9. \boxed{3}"
```

**Why?** This teaches the model:
- How to think step-by-step
- How to format the answer
- What level of detail to provide

### 🔄 The Training Process

#### Phase 1: Problem Selection
```
Batch 1: Problems 1-128
Batch 2: Problems 129-256
Batch 3: Problems 257-384
... and so on
```

#### Phase 2: Group Sampling

For each problem, generate multiple attempts:

```
Problem: "What is 7 × 8?"

Attempt 1: "7 × 8 = 56. \boxed{56}" ✅ Reward: 1.0
Attempt 2: "7 + 8 = 15. \boxed{15}" ❌ Reward: 0.0
Attempt 3: "7 × 8 = 54. \boxed{54}" ❌ Reward: 0.0
Attempt 4: "7 × 8 = 56. \boxed{56}" ✅ Reward: 1.0
... (12 more attempts)

Average reward: 0.25
Advantages: [+0.75, -0.25, -0.25, +0.75, ...]
```

#### Phase 3: Learning

**Model learns to:**
- Recognize problem patterns
- Apply correct mathematical operations
- Show work clearly
- Format answers properly

### 📊 Different Dataset Characteristics

#### GSM8K (Easy)
- **Level:** Elementary math
- **Skills:** Basic arithmetic, word problems
- **Training time:** Fast (1 min/iteration)
- **Expected accuracy:** 70-80% after training

#### MATH (Hard)
- **Level:** High school competition
- **Skills:** Algebra, geometry, calculus
- **Training time:** Moderate (2 min/iteration)
- **Expected accuracy:** 40-50% after training

#### Polaris/DeepMath (Very Hard)
- **Level:** University mathematics
- **Skills:** Advanced topics, proofs
- **Training time:** Slow (3-4 min/iteration)
- **Expected accuracy:** 20-30% after training

### 💡 Key Insights

**1. The Grading System is Smart:**
```
Student writes: \boxed{1/2}
Correct answer: \boxed{0.5}

Grader says: ✅ CORRECT! (It knows 1/2 = 0.5)
```

The grader uses symbolic math to compare answers, not just string matching!

**2. Timeouts Prevent Cheating:**

If the grader takes too long (>1 second), the answer is marked wrong. This prevents the model from writing overly complex expressions that are technically correct but unusable.

**3. Format Matters:**

```
Without format check:
Model writes: "The answer is 42" → Can't extract answer → 0 reward

With format check:
Model writes: "The answer is \boxed{42}" → Clear answer → 1.0 reward
```

### 🎓 What the Model Actually Learns

After training, the model develops:

1. **Problem Classification:**
   - "This is an addition problem"
   - "This needs algebra"
   - "This is a word problem about ratios"

2. **Solution Strategies:**
   - "Break complex problems into steps"
   - "Check if my answer makes sense"
   - "Show my work clearly"

3. **Format Awareness:**
   - "Always put final answer in \boxed{}"
   - "Explain my reasoning"
   - "Keep it concise"

---

## 4. Code RL - Training for Code Generation

### 📝 What This Recipe Does

**Goal:** Train a model to write code that passes test cases

**Task:** Given a programming problem and tests, write working code

**Example Problem:**
```
"Write a function that takes a list of numbers and returns 
the sum of all even numbers."

Tests:
- sum_evens([1, 2, 3, 4]) → should return 6
- sum_evens([]) → should return 0
- sum_evens([1, 3, 5]) → should return 0
```

### 🏗️ The Architecture

#### Three Main Components:

**1. The Sandbox (Safety First!)**

Think of it like a virtual computer that can be reset:
- Model writes code
- Code runs in isolated environment
- If code crashes/hangs → Sandbox resets
- Main system stays safe

**Two sandbox options:**
- **SandboxFusion:** Local Docker container
- **Modal:** Cloud-based sandbox

**Why needed?** Without a sandbox, bad code could:
- Delete files
- Use infinite loops (crash your computer)
- Access sensitive data

**2. The Problem Dataset**

**Sources:**
- PrimeIntellect coding problems
- TACO coding challenges  
- LiveCodeBench competitions
- Codeforces problems

**Problem format:**
```python
{
  "question": "Write a function to find the maximum...",
  "tests": [
    {"input": "[1,5,3]", "output": "5"},
    {"input": "[10,2,8]", "output": "10"}
  ],
  "starter_code": "def find_max(arr):\n    # Your code here"
}
```

**3. The Reward System**

More sophisticated than math problems!

```python
Reward breakdown:
- Syntax correct: +0.2
- Tests passed: +0.6 per test
- All tests passed: +1.0
- Runtime error: -0.5
- Timeout: -0.5
```

### 🔄 The Training Process (Detailed Example)

#### Problem Presented:

```
"Write a function that reverses a string."

Tests:
- reverse("hello") → "olleh"
- reverse("") → ""
- reverse("a") → "a"
```

#### Model Generates 8 Solutions:

**Solution 1:**
```python
def reverse(s):
    return s[::-1]
```
**Execution:** Runs in sandbox → All tests pass! ✅
**Reward:** +1.0

**Solution 2:**
```python
def reverse(s):
    result = ""
    for char in s:
        result = char + result
    return result
```
**Execution:** All tests pass! ✅
**Reward:** +1.0

**Solution 3:**
```python
def reverse(s):
    return s.reverse()
```
**Execution:** Error! strings don't have .reverse() method ❌
**Reward:** -0.5

**Solution 4:**
```python
def reverse(s):
    return reversed(s)
```
**Execution:** Returns iterator, not string ❌
**Tests:** Fail ❌
**Reward:** 0.0

**Solution 5:**
```python
def reverse(s):
    list_s = list(s)
    list_s.reverse()
    return ''.join(list_s)
```
**Execution:** All tests pass! ✅
**Reward:** +1.0

... (3 more attempts)

**Advantages Computed:**
```
Mean reward = (1.0 + 1.0 - 0.5 + 0.0 + 1.0 + ...) / 8 = 0.3125

Solution 1: 1.0 - 0.3125 = +0.6875 (great!)
Solution 2: 1.0 - 0.3125 = +0.6875 (great!)
Solution 3: -0.5 - 0.3125 = -0.8125 (bad!)
Solution 4: 0.0 - 0.3125 = -0.3125 (below average)
Solution 5: 1.0 - 0.3125 = +0.6875 (great!)
```

#### Model Learns:

**Reinforce (do more):**
- Using slicing `s[::-1]`
- Loop with concatenation
- Using `list()` and `reverse()` method

**Discourage (do less):**
- Using non-existent string methods
- Returning wrong types
- Approaches that error out

### 🎯 Advanced Features

#### 1. Test Type Handling

**Two test formats:**

**stdin/stdout (Input/Output)**
```python
# Problem runs code with input, checks output
Input: "5\n3\n"
Expected output: "8\n"
```

**Functional (Unit tests)**
```python
# Problem calls function directly
assert add(5, 3) == 8
assert add(0, 0) == 0
```

#### 2. Starter Code

Some problems give you a template:

```python
Given:
def process_data(data):
    # TODO: implement this
    pass

Model must fill in the logic while keeping the signature.
```

#### 3. LiveCodeBench Integration

Uses real competitive programming contest problems:
- Problems from 2024-2025 contests
- Tests pass rate benchmarks
- Tracks improvement over time

### 📊 What Success Looks Like

**Before Training:**
- Pass@1 (best of 1): ~34%
- Pass@8 (best of 8): ~44%

**After 100 iterations:**
- Pass@1: ~43% (+9 points!)
- Pass@8: ~55% (+11 points!)

**What this means:**
- Model gets better at first-try solutions
- Model explores more valid approaches
- Model avoids common errors

### 💡 What the Model Actually Learns

**1. Syntax Patterns:**
- "Use `[::-1]` for reversing"
- "Remember to return, not just print"
- "List comprehensions are powerful"

**2. Debugging Strategies:**
- "If it errors, try a different approach"
- "Test edge cases (empty input, size 1)"
- "Be careful with types (string vs list)"

**3. Code Structure:**
- "Handle base cases first"
- "Use clear variable names"
- "One logical step at a time"

### 🚨 Safety Note

**Why Sandboxing is Critical:**

Without sandbox, malicious/buggy code could:
```python
# Bad code the model might generate:
while True:
    pass  # Infinite loop - freezes computer

import os
os.system("rm -rf /")  # Deletes everything!

open("important.txt", "w").write("")  # Overwrites files
```

With sandbox:
- Code runs in isolation
- Timeout kills infinite loops (2 seconds)
- No access to your real files
- If it crashes → just restart sandbox

---

## 5. Multiplayer RL - Multi-Agent Training

### 📝 What This Recipe Does

**Goal:** Train models that interact with other agents (human or AI)

**Three Progressive Examples:**

1. **Guess the Number** (Simplest)
2. **Twenty Questions** (Moderate)
3. **Text Arena / Tic-Tac-Toe** (Advanced)

Let me explain each one in detail:

---

### 🎲 Example 1: Guess the Number

#### The Game

**Setup:**
- Computer picks a secret number (1-100)
- AI tries to guess it
- Computer says "higher" or "lower"
- AI gets 10 tries maximum

#### Sample Game:

```
Computer: "I'm thinking of a number between 1 and 100."

AI: "Is it 50?"
Computer: "Higher!"

AI: "Is it 75?"
Computer: "Lower!"

AI: "Is it 62?"
Computer: "Higher!"

AI: "Is it 68?"
Computer: "Lower!"

AI: "Is it 65?"
Computer: "Correct! You got it in 5 guesses!"
```

#### The Training Process

**Reward Structure:**
```
Found in 1 guess: +10.0 (lucky!)
Found in 2 guesses: +9.0
Found in 3 guesses: +8.0
...
Found in 10 guesses: +1.0
Failed to find: 0.0
```

**What Happens During Training:**

**Early Training (Random guessing):**
```
Attempt 1: "Is it 37?" → Lower → "Is it 82?" → Lower → "Is it 11?" → ...
Result: Takes 8-10 guesses, often fails
Average reward: +2.0
```

**Mid Training (Learning patterns):**
```
Attempt 1: "Is it 50?" → Higher → "Is it 75?" → Lower → "Is it 62?" → ...
Result: Takes 5-7 guesses
Average reward: +5.0
```

**Late Training (Binary search!):**
```
Attempt 1: "Is it 50?" → Higher → "Is it 75?" → Higher → "Is it 87?" → ...
Result: Takes 3-4 guesses
Average reward: +7.5
```

**What the AI Discovers:**

The AI doesn't know "binary search" initially, but learns:
1. "Start in the middle"
2. "Each guess should eliminate half the possibilities"
3. "Never guess outside the current range"

**Why This is Amazing:**

The AI **invents binary search from scratch** just by trying to maximize rewards!

---

### 🤔 Example 2: Twenty Questions

#### The Game

**Setup:**
- Computer picks a secret word (e.g., "elephant")
- AI asks yes/no questions
- Computer answers truthfully
- AI gets 20 questions
- AI must guess the word

#### Sample Game:

```
Secret word: ELEPHANT

AI: "Is it alive?"
Computer: "Yes."

AI: "Is it a plant?"
Computer: "No."

AI: "Is it an animal?"
Computer: "Yes."

AI: "Is it a mammal?"
Computer: "Yes."

AI: "Does it live in water?"
Computer: "No."

AI: "Is it larger than a human?"
Computer: "Yes."

AI: "Does it have a trunk?"
Computer: "Yes."

AI: "Is it an elephant?"
Computer: "Yes! You got it in 8 questions!"
```

#### What Makes This Harder

**Challenge:** The "computer" is actually another AI!

**Two AI systems:**
1. **Guesser (training):** Learns to ask good questions
2. **Answerer (fixed):** Llama-3.1-8B-Instruct, answers questions

**Why harder:** The guesser must learn to:
- Ask questions that narrow down possibilities
- Remember what it already learned
- Build on previous answers
- Make logical deductions

#### The Training Process

**Reward Structure:**
```
Guessed in 1-5 questions: +10.0 (amazing!)
Guessed in 6-10 questions: +7.0 (good)
Guessed in 11-15 questions: +4.0 (okay)
Guessed in 16-20 questions: +1.0 (barely)
Failed to guess: 0.0
```

**Training Phases:**

**Phase 1: Random questions (bad)**
```
Q1: "Is it blue?" → "No."
Q2: "Can it fly?" → "No."
Q3: "Is it made of metal?" → "No."
... (asking disconnected questions)
Result: Rarely guesses correctly
```

**Phase 2: Developing strategy (better)**
```
Q1: "Is it living?" → "Yes."
Q2: "Is it an animal?" → "Yes."
Q3: "Is it a pet?" → "No."
Q4: "Does it live in the wild?" → "Yes."
... (following a logical path)
Result: Sometimes guesses correctly
```

**Phase 3: Optimized questioning (best)**
```
Q1: "Is it alive?" → "Yes."
Q2: "Is it an animal?" → "Yes."
Q3: "Is it a mammal?" → "Yes."
Q4: "Is it larger than a car?" → "Yes."
Q5: "Does it have a trunk?" → "Yes."
Q6: "Is it an elephant?" → "Yes!"
Result: Usually guesses within 10 questions
```

**What the Guesser Learns:**

1. **Taxonomy questioning:**
   - Start broad: "Is it alive?"
   - Get more specific: "Is it a mammal?"
   - Zero in: "Does it have stripes?"

2. **Information theory:**
   - "Questions that split possibilities 50/50 are best"
   - "Don't ask about rare features early"
   - "Build on what I already know"

3. **Deductive reasoning:**
   - "If it's an animal and large and has trunk → probably elephant"
   - "If it's not alive and made of wood → probably furniture"

---

### 🎮 Example 3: Text Arena (Tic-Tac-Toe)

#### The Game

**Setup:**
- Two AI players play Tic-Tac-Toe
- Each AI controls X or O
- AIs learn by playing each other
- Goal: Win games, avoid losses

#### Sample Game:

```
Board:    1|2|3
          4|5|6
          7|8|9

AI-X: "I place X in position 5 (center)"
Board:    . | . | .
          . | X | .
          . | . | .

AI-O: "I place O in position 1 (corner)"
Board:    O | . | .
          . | X | .
          . | . | .

AI-X: "I place X in position 9 (opposite corner)"
Board:    O | . | .
          . | X | .
          . | . | X

AI-O: "I place O in position 3 (blocking corner)"
Board:    O | . | O
          . | X | .
          . | . | X

... game continues ...
```

#### What Makes This MOST Complex

**Challenge:** Both players are learning simultaneously!

**Self-Play Dynamics:**
- AI-X gets better → AI-O must adapt
- AI-O finds new strategy → AI-X must counter
- Both players improve together
- Creates an "arms race" of strategies

#### The Training Process

**Reward Structure:**
```
Win: +1.0
Draw: +0.0
Loss: -1.0
```

**Training Phases:**

**Phase 1: Random play**
```
Both AIs place pieces randomly
Win rate: ~50/50 (whoever gets lucky)
Many draws
Strategy: None
```

**Phase 2: Learning basics**
```
AIs discover:
- "Three in a row wins"
- "Block opponent's winning moves"
- "Center is often good"
Win rate: Still ~50/50 but fewer random draws
```

**Phase 3: Advanced strategy**
```
AIs develop:
- "Corners are strong opening moves"
- "Create fork opportunities" (two ways to win)
- "Recognize trap patterns"
Win rate: Approaches optimal play (mostly draws between skilled players)
```

**Phase 4: Near-optimal play**
```
Both AIs play almost perfectly
Most games end in draws (as they should in optimal play)
Rare wins come from opponent mistakes
```

#### What Makes Self-Play Powerful

**The Magic of Co-Evolution:**

```
Day 1: Both AIs are terrible
  → They play random games
  → But even bad data teaches basics

Day 5: AIs learn basic rules
  → They start making sensible moves
  → Training gets harder (opponent is better!)
  
Day 10: AIs develop strategies
  → Each AI finds weaknesses in opponent
  → Forces opponent to improve that weakness
  → Creates positive feedback loop

Day 20: Near-optimal play
  → Both AIs play like experts
  → Training curve flattens (near perfect play)
```

**Why This Works:**

Imagine learning chess:
- **Playing vs beginner:** You learn basic rules
- **Playing vs intermediate:** You learn strategy
- **Playing vs expert:** You learn advanced tactics

By training two AIs together, they provide each other with opponents that are always at the perfect difficulty level - challenging but not impossible!

---

## 🎓 Key Takeaways Across All Recipes

### Common Patterns

**1. The Core Loop Never Changes:**
```
Generate → Evaluate → Learn → Repeat
```

**2. Complexity Comes from Environment:**
- Math: Simple (one response, check answer)
- Code: Moderate (run code, check tests)
- Multiplayer: Complex (multiple turns, other agents)

**3. Reward Design is Critical:**
- Too sparse → Model doesn't learn
- Too dense → Model exploits loopholes
- Just right → Model learns desired behavior

### What Models Actually Learn

**Not Memorization:**
- Models don't memorize answers
- They learn **strategies** and **patterns**

**Emergent Behaviors:**
- Binary search (Guess the Number)
- Logical questioning (Twenty Questions)  
- Game theory (Tic-Tac-Toe)

**Generalization:**
- Trained on easy problems → Can solve harder ones
- Learns principles, not specific cases

### Why RL is Powerful

**Advantages:**
1. **No need for perfect solutions** - Just need a way to score
2. **Discovers novel approaches** - Not limited to training data
3. **Adapts to environment** - Learns what works
4. **Handles complex tasks** - Multi-turn, multi-agent scenarios

**Limitations:**
1. **Requires more compute** - Multiple samples per problem
2. **Can be unstable** - Need careful hyperparameter tuning
3. **Reward design is tricky** - Hard to specify exactly what you want

---

## 🚀 Progression Path

If you're learning RL, go in this order:

```
1. RL Basic (math-rl)
   ↓
   Learn: Basic RL loop, rewards, advantages
   
2. RL Loop (simplified)
   ↓
   Learn: What's essential vs extra features
   
3. Code RL
   ↓
   Learn: Verification, sandboxing, complex rewards
   
4. Guess Number (multiplayer)
   ↓
   Learn: Multi-turn environments
   
5. Twenty Questions (multiplayer)
   ↓
   Learn: Agent interaction, multi-agent
   
6. Tic-Tac-Toe (multiplayer)
   ↓
   Learn: Self-play, co-evolution
```

---

## 💬 Final Thoughts

**The Beauty of RL:**

These recipes show that you don't need to be an expert in:
- Math (to train a math solver)
- Programming (to train a code generator)
- Game theory (to train a game player)

You just need:
1. A way to present problems
2. A way to evaluate attempts
3. Patience to let the model explore

The AI figures out the rest through trial and error!

**Remember:** Even complex behaviors (like binary search or game strategy) emerge naturally when the AI is given:
- Clear goals (rewards)
- Freedom to explore (sampling)
- Feedback on performance (advantages)

That's the magic of reinforcement learning! ✨

