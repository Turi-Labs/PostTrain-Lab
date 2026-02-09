# PostTrain-Lab Documentation

Welcome to the documentation hub! Here you'll find everything you need to understand RL for LLMs.

## 📚 Documentation Guide

### 🎯 Start Here (For Beginners)

**1. [RL_EXPLAINED.md](./RL_EXPLAINED.md)** - Your complete guide
   - What is RL for LLMs? (with analogies!)
   - Different RL methods explained
   - Step-by-step pipeline walkthrough
   - Code examples with plain English explanations
   - All the different ways to do RL
   - **Time to read:** 45-60 minutes
   - **Best for:** Understanding concepts deeply

**2. [RL_VISUAL_SUMMARY.md](./RL_VISUAL_SUMMARY.md)** - Visual reference
   - ASCII diagrams of how RL works
   - Learning curves and what they mean
   - Reward structures visualized
   - Hyperparameter impact charts
   - **Time to read:** 15-20 minutes
   - **Best for:** Quick visual understanding

### 📖 Deep Dives

**3. [RL_RECIPES_EXPLAINED.md](./RL_RECIPES_EXPLAINED.md)** - Recipe walkthroughs
   - What each Tinker recipe actually does
   - Explained without assuming you know code
   - Step-by-step breakdowns of:
     - RL Basic (Math problems)
     - RL Loop (Minimal example)
     - Math RL (Multiple datasets)
     - Code RL (Code generation)
     - Multiplayer RL (Games and multi-agent)
   - **Time to read:** 60-90 minutes
   - **Best for:** Understanding real implementations

### ⚡ Quick Reference

**4. [RL_QUICK_REFERENCE.md](./RL_QUICK_REFERENCE.md)** - Cheat sheet
   - The 5-step core loop
   - Essential code patterns
   - Hyperparameter cheat sheet
   - Common mistakes to avoid
   - Debugging checklist
   - **Time to read:** 10 minutes
   - **Best for:** When you need to look something up fast

---

## 🎓 Learning Paths

### Path 1: Complete Beginner (Never heard of RL)

```
Day 1: Read RL_EXPLAINED.md (sections 1-3)
       → Understand: What is RL, Why use it

Day 2: Read RL_VISUAL_SUMMARY.md
       → See: Visual diagrams, learning curves

Day 3: Read RL_EXPLAINED.md (sections 4-5)
       → Understand: Code walkthrough, Tinker specifics

Day 4: Read RL_RECIPES_EXPLAINED.md (RL Basic only)
       → See: First real example

Day 5: Bookmark RL_QUICK_REFERENCE.md
       → Ready to start experimenting!
```

### Path 2: Some ML Background

```
Day 1: Read RL_EXPLAINED.md (skim familiar parts)
       → Focus on: RL-specific concepts

Day 2: Read RL_RECIPES_EXPLAINED.md (all recipes)
       → Understand: Real-world applications

Day 3: Read RL_QUICK_REFERENCE.md
       → Keep handy: For implementation
```

### Path 3: Experienced ML Engineer

```
Hour 1: Skim RL_EXPLAINED.md (code sections)
        → Get: Tinker API patterns

Hour 2: Read RL_QUICK_REFERENCE.md
        → Reference: Code patterns, hyperparams

Hour 3: Browse RL_RECIPES_EXPLAINED.md
        → Find: Relevant use cases

Ready to implement!
```

---

## 🎯 Find What You Need

### I want to understand...

**"What is RL for LLMs?"**
→ Go to: [RL_EXPLAINED.md - Section 1](./RL_EXPLAINED.md#what-is-rl-for-llms)

**"How does the training loop work?"**
→ Go to: [RL_EXPLAINED.md - Section 3](./RL_EXPLAINED.md#step-by-step-pipeline)

**"What's the difference between RLVR and RLHF?"**
→ Go to: [RL_EXPLAINED.md - Section 2](./RL_EXPLAINED.md#different-rl-methods)

**"How do advantages work?"**
→ Go to: [RL_VISUAL_SUMMARY.md - Advantages](./RL_VISUAL_SUMMARY.md#advantage-computation-explained)

**"What's PPO vs importance sampling?"**
→ Go to: [RL_EXPLAINED.md - Loss Functions](./RL_EXPLAINED.md#-loss-functions)

### I want to implement...

**"My first RL training"**
→ Go to: [RL_RECIPES_EXPLAINED.md - RL Basic](./RL_RECIPES_EXPLAINED.md#1-rl-basic---your-first-rl-run)

**"Code generation with RL"**
→ Go to: [RL_RECIPES_EXPLAINED.md - Code RL](./RL_RECIPES_EXPLAINED.md#4-code-rl---training-for-code-generation)

**"A multi-turn environment"**
→ Go to: [RL_RECIPES_EXPLAINED.md - Multiplayer](./RL_RECIPES_EXPLAINED.md#5-multiplayer-rl---multi-agent-training)

**"Check code patterns"**
→ Go to: [RL_QUICK_REFERENCE.md](./RL_QUICK_REFERENCE.md)

### I'm debugging...

**"Training is unstable"**
→ Go to: [RL_QUICK_REFERENCE.md - Debugging](./RL_QUICK_REFERENCE.md#-debugging-checklist)

**"Pass rate not improving"**
→ Go to: [RL_VISUAL_SUMMARY.md - Training Progress](./RL_VISUAL_SUMMARY.md#-training-progress-what-youll-see)

**"Don't understand error message"**
→ Go to: [RL_QUICK_REFERENCE.md - Common Mistakes](./RL_QUICK_REFERENCE.md#-common-mistakes)

---

## 📊 Documentation Map

```
docs/
│
├── README.md (You are here!)
│   └── Navigation guide
│
├── RL_EXPLAINED.md ⭐ START HERE
│   ├── What is RL?
│   ├── Different methods
│   ├── Step-by-step pipeline
│   ├── Code walkthrough
│   └── Tinker concepts
│
├── RL_VISUAL_SUMMARY.md 📊 VISUAL LEARNER
│   ├── Diagrams
│   ├── Charts
│   ├── Learning curves
│   └── Quick visual reference
│
├── RL_RECIPES_EXPLAINED.md 📖 DEEP DIVE
│   ├── RL Basic explained
│   ├── RL Loop explained
│   ├── Math RL explained
│   ├── Code RL explained
│   └── Multiplayer RL explained
│
└── RL_QUICK_REFERENCE.md ⚡ CHEAT SHEET
    ├── 5-step loop
    ├── Code patterns
    ├── Hyperparameters
    ├── Common mistakes
    └── Debugging tips
```

---

## 🎯 Key Concepts Overview

### The Core Idea

```
RL = Learning from trial and error

Instead of showing correct answers (SFT):
1. Let model try different solutions
2. Score each attempt
3. Do more of what works
4. Repeat until good
```

### The Core Loop

```
1. SAMPLE    → Generate multiple solutions
2. EVALUATE  → Score each one
3. ADVANTAGE → Compare to average
4. PREPARE   → Create training data
5. TRAIN     → Update model
```

### The Magic

```
Model learns STRATEGIES, not memorization:
• How to solve problems
• What approaches work
• How to avoid mistakes
• When to try different methods
```

---

## 💡 Quick Tips

### Before You Start Reading

1. **Don't rush** - RL is conceptually simple but has many details
2. **Read in order** - Each doc builds on previous ones
3. **Take breaks** - Let concepts sink in
4. **Try examples** - Best way to understand is to run code
5. **Ask questions** - Revisit sections that confuse you

### While Reading

- **See a term you don't know?** Check RL_QUICK_REFERENCE.md
- **Confused by a concept?** Look for the visual in RL_VISUAL_SUMMARY.md
- **Want more detail?** Check the corresponding recipe in RL_RECIPES_EXPLAINED.md
- **Need code example?** Look in RL_EXPLAINED.md code sections

### After Reading

- **Test your understanding** - Can you explain the 5-step loop to someone?
- **Try a recipe** - Run RL Basic and watch it learn
- **Experiment** - Change one hyperparameter, see what happens
- **Read again** - Second read-through will click more pieces together

---

## 🚀 What's Next?

After reading these docs, you're ready to:

1. **Run the examples** in `methods/rlvr/`
2. **Try Tinker recipes** in `tinker-cookbook/recipes/`
3. **Experiment** with different hyperparameters
4. **Build** your own RL training pipeline
5. **Share** what you learn!

---

## 📬 Feedback

Found something confusing? Want more examples? Have suggestions?

These docs are meant to help YOU understand RL. Let us know how to make them better!

---

## 🎓 Remember

```
┌──────────────────────────────────────────────┐
│                                              │
│  "The journey of learning RL is like the    │
│   RL process itself:                        │
│                                              │
│   Try things → See what works →             │
│   Learn from it → Try again                 │
│                                              │
│   You'll get better each iteration!"        │
│                                              │
└──────────────────────────────────────────────┘
```

**Happy Learning! 🎉**

