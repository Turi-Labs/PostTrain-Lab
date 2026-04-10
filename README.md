# PostTrain-Lab

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A personal research lab for exploring, implementing, and eventually inventing post-training techniques for LLMs.

Everything runs on [Tinker](https://thinkingmachines.ai/) — a hosted training API that handles the GPU side, so experiments stay clean Python notebooks.

---

## The Goal

Post-training is where a pre-trained model becomes actually useful. This lab exists to:

1. **Master the foundation** — implement every known post-training methodology from scratch, understand what each one does and when it works
2. **Invent new pipelines** — once the fundamentals are solid, combine and extend them in ways that haven't been tried

---

## Post-Training Methodologies

There are 4 core approaches. Everything else (math RL, code RL, tool use RL) is just one of these with a different dataset or reward signal.

### 1. Supervised Fine-Tuning (SFT)
Standard next-token prediction on labeled data. The simplest and most common form of post-training. Teaches a model new behaviors by showing it examples.

### 2. Reinforcement Learning (RL)
The model generates outputs, and a reward signal tells it what was good. The reward can come from anything — a verifier, a code executor, an LLM judge, a math checker. The model learns to maximize that reward.

### 3. Preference Learning
Align the model with what humans (or AI) prefer, rather than just labelled examples.
- **DPO** — learns directly from (chosen, rejected) pairs. No reward model needed.
- **RLHF** — first trains a reward model on preferences, then runs RL against it.

### 4. Distillation
Transfer knowledge from a stronger teacher model into the student.
- **On-policy** — student generates, teacher scores/corrects, student learns from that
- **Off-policy** — student learns from pre-collected teacher outputs
- **SDFT** (Self-Distillation) — no teacher needed; model distills from itself using forward KL

---

## Roadmap

### Phase 1 — Foundation

Implement every major methodology as a clean, runnable notebook. Each notebook should be self-contained: dataset, training, and a clear before/after comparison.

| Method | Notebook | Status |
|---|---|---|
| Supervised Fine-Tuning (SFT) | `methods/sft/sft_train.ipynb` | ✅ Done |
| RL with Verifiable Rewards (RLVR) | `methods/rlvr/rlvr_train.ipynb` | ✅ Done |
| Direct Preference Optimization (DPO) | `methods/dpo/` | 📋 Planned |
| RLHF (reward model + RL) | `methods/rlhf/` | 📋 Planned |
| On-policy Distillation | `methods/distillation/` | 📋 Planned |
| Rubric-based RL | `methods/rubric_rl/` | 📋 Planned |
| Self-Distillation (SDFT) | `methods/sdft/` | 📋 Planned |
| Multi-Agent RL | `methods/multiagent_rl/` | 📋 Planned |

### Phase 2 — Invention

Once the foundation is solid, the interesting questions:

- **SFT → RL**: bootstrap with SFT, continue with RL. How much does warm-starting help?
- **Distill → RL**: distill from a strong teacher, then RL to push the student beyond the teacher
- **Multi-objective RL**: combine multiple reward signals (correctness + conciseness + style)
- **Curriculum RL**: progressively harder training problems vs. flat distribution
- **Iterative RLHF**: train reward model → run RL → collect new preferences → retrain reward model → repeat
- **Cross-method comparison**: same task, same model — SFT vs DPO vs RL. When does each win?

---

## Project Structure

```
PostTrain-Lab/
├── methods/                # One folder per methodology
│   ├── sft/               # Supervised Fine-Tuning
│   ├── rlvr/              # RL with Verifiable Rewards
│   ├── dpo/               # Direct Preference Optimization
│   ├── rlhf/              # RLHF pipeline
│   ├── distillation/      # Knowledge Distillation
│   ├── rubric_rl/         # Rubric-based RL
│   ├── sdft/              # Self-Distillation Fine-Tuning
│   └── multiagent_rl/     # Multi-Agent RL
│
└── tinker-cookbook/        # Reference implementations (from Thinking Machines)
```

---

## Getting Started

```bash
git clone https://github.com/yourusername/PostTrain-Lab.git
cd PostTrain-Lab
pip install -r requirements.txt
export TINKER_API_KEY=<your-key>
```

Start with SFT:

```bash
jupyter notebook methods/sft/sft_train.ipynb
```

---

## Infrastructure

Built on [Tinker](https://thinkingmachines.ai/) — training loops run locally in Python, GPU work runs remotely via the Tinker API.

---

## Acknowledgments

- [Thinking Machines](https://thinkingmachines.ai/) for the $5k grant
- The [tinker-cookbook](./tinker-cookbook/) for reference implementations
