# Phase 1 Plan — PostTrain-Lab

The goal of Phase 1 is to implement every major post-training methodology as a clean, runnable notebook. Each experiment should be self-contained: a dataset, a training loop, and a clear before/after comparison so the effect of the method is observable.

All training runs on [Tinker](https://thinkingmachines.ai/). Reference implementations live in `tinker-cookbook/`.

---

## A Note on PPO

PPO (Proximal Policy Optimization) is the **algorithm** that powers most RL-based post-training — it shows up in RLVR, Rubric RL, and RLHF. It is not a separate methodology. You've already used it in RLVR. No dedicated notebook needed.

---

## Experiment Sequence

### 1. SFT — Supervised Fine-Tuning
**Status:** ✅ Done  
**Notebook:** `methods/sft/sft_train.ipynb`  
**What it does:** Next-token prediction on labeled examples. The baseline for all other methods.  
**Reference:** `tinker-cookbook/tinker_cookbook/recipes/chat_sl/`

---

### 2. RLVR — RL with Verifiable Rewards
**Status:** ✅ Done  
**Notebook:** `methods/rlvr/rlvr_train.ipynb`  
**What it does:** Model generates outputs, a verifier (code executor / math checker) gives binary reward. Uses PPO to reinforce correct outputs.  
**Reference:** `tinker-cookbook/tinker_cookbook/recipes/code_rl/`, `math_rl/`

---

### 3. DPO — Direct Preference Optimization
**Status:** 📋 Planned  
**Notebook:** `methods/dpo/dpo_train.ipynb`  
**What it does:** Given (chosen, rejected) response pairs, directly optimize the model to prefer the chosen response. No reward model needed.  
**Key idea:** Reformulates the RLHF objective into a supervised loss — simpler and more stable than PPO-based preference learning.  
**Dataset:** A preference dataset (e.g., HH-RLHF or UltraFeedback)  
**Reference:** `tinker-cookbook/tinker_cookbook/recipes/preference/dpo/`

---

### 4. Rubric RL — RL with LLM-as-Judge
**Status:** 📋 Planned  
**Notebook:** `methods/rubric_rl/rubric_rl_train.ipynb`  
**What it does:** Like RLVR, but the reward signal comes from an LLM judge using a rubric, not a code executor. Opens RL up to tasks that can't be verified programmatically (writing quality, reasoning style, etc.).  
**Key idea:** Bridges RL and LLM-as-judge evaluation. The judge's rubric is the reward function.  
**Dataset:** Any open-ended task with evaluable outputs (summarization, reasoning, instruction following)  
**Reference:** `tinker-cookbook/tinker_cookbook/recipes/rubric/`

---

### 5. RLHF — Reinforcement Learning from Human Feedback
**Status:** 📋 Planned  
**Notebook:** `methods/rlhf/rlhf_train.ipynb`  
**What it does:** Full 3-stage pipeline:
  1. SFT warm-start
  2. Train a reward model on preference data
  3. Run RL (PPO) against the reward model  

**Key idea:** Unlike DPO, RLHF learns an explicit reward model first. This is more flexible (the reward model can generalize) but heavier to train.  
**Dependency:** Builds on SFT (stage 1) and preference data (stage 2)  
**Reference:** `tinker-cookbook/tinker_cookbook/recipes/preference/rlhf/`

---

### 6. On-Policy Distillation
**Status:** 📋 Planned  
**Notebook:** `methods/distillation/on_policy.ipynb`  
**What it does:** The student model generates outputs; the teacher scores or corrects them; the student trains on teacher-guided signal. Student and teacher interact at training time.  
**Key idea:** The student learns from its own distribution, corrected by the teacher — avoids distribution mismatch of off-policy methods.  
**Reference:** `tinker-cookbook/tinker_cookbook/recipes/distillation/on_policy_distillation.py`

---

### 7. Off-Policy Distillation
**Status:** 📋 Planned  
**Notebook:** `methods/distillation/off_policy.ipynb`  
**What it does:** The student trains on pre-collected outputs from the teacher. No live teacher at training time.  
**Key idea:** Cheaper and simpler than on-policy. Useful when teacher inference is expensive. Suffers from distribution mismatch as the student improves.  
**Reference:** `tinker-cookbook/tinker_cookbook/recipes/distillation/off_policy_reasoning.py`

---

### 8. SDFT — Self-Distillation Fine-Tuning
**Status:** 📋 Planned  
**Notebook:** `methods/sdft/sdft_train.ipynb`  
**What it does:** No external teacher. The model distills from itself using a forward KL loss, selecting its top-K generations to learn from.  
**Key idea:** Gets distillation-style benefits without a stronger teacher. Works by having the model reinforce its own best outputs.  
**Reference:** `tinker-cookbook/tinker_cookbook/recipes/sdft/`

---

### 9. MARL — Multi-Agent Reinforcement Learning
**Status:** 📋 Planned  
**Notebook:** `methods/marl/marl_train.ipynb`  
**What it does:** Two model instances play against each other (or one plays with itself). Reward comes from winning or outperforming the opponent. The model improves through self-play.  
**Key idea:** Reward signal emerges from competition, not a fixed verifier or judge. Useful for tasks with natural adversarial structure (debate, games, negotiation).  
**Dataset:** A game or adversarial task (e.g., 20 Questions, Number Guessing)  
**Reference:** `tinker-cookbook/tinker_cookbook/recipes/multiplayer_rl/`

---

## Summary Table

| # | Method | Type | Status | Notebook |
|---|---|---|---|---|
| 1 | SFT | Supervised | ✅ Done | `methods/sft/sft_train.ipynb` |
| 2 | RLVR | RL | ✅ Done | `methods/rlvr/rlvr_train.ipynb` |
| 3 | DPO | Preference | 📋 Planned | `methods/dpo/dpo_train.ipynb` |
| 4 | Rubric RL | RL | 📋 Planned | `methods/rubric_rl/rubric_rl_train.ipynb` |
| 5 | RLHF | Preference + RL | 📋 Planned | `methods/rlhf/rlhf_train.ipynb` |
| 6 | On-Policy Distillation | Distillation | 📋 Planned | `methods/distillation/on_policy.ipynb` |
| 7 | Off-Policy Distillation | Distillation | 📋 Planned | `methods/distillation/off_policy.ipynb` |
| 8 | SDFT | Distillation | 📋 Planned | `methods/sdft/sdft_train.ipynb` |
| 9 | MARL | RL | 📋 Planned | `methods/marl/marl_train.ipynb` |

---

## What Each Notebook Should Contain

Every notebook follows the same structure:
1. **Concept** — what this method is and why it exists (5 min read)
2. **Setup** — dataset loading, model config, Tinker client
3. **Training loop** — the actual implementation
4. **Results** — before/after metrics, loss curves
5. **Key observations** — what worked, what didn't, what to try next

---

## Phase 2 Preview

Once all 9 notebooks are done, Phase 2 experiments will combine and extend these methods:
- SFT → RL warm-starting
- Distill → RL (push student beyond teacher)
- Multi-objective RL (combine reward signals)
- Iterative RLHF (retrain reward model in a loop)
- Cross-method benchmarks (same task, all methods, who wins?)
