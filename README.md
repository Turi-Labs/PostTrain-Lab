# 🧠 PostTrain-Lab

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive, hands-on laboratory for exploring post-training techniques for Large Language Models

PostTrain-Lab is a modular educational project designed to demystify the art and science of post-training LLMs. From supervised fine-tuning to cutting-edge reinforcement learning techniques, this repository provides a unified playground to experiment, learn, and compare different approaches.

## 🎯 What is Post-Training?

Post-training refers to the techniques applied to pre-trained language models to adapt them for specific tasks, align them with human preferences, or improve their capabilities. This includes:

- **Supervised Fine-Tuning (SFT)** - Teaching models specific tasks with labeled examples
- **Preference Learning (DPO, RLHF)** - Aligning models with human preferences
- **Reinforcement Learning (RLAIF, RLVR)** - Optimizing models through reward signals
- **Continual Learning** - Updating models without forgetting previous knowledge
- **Distillation** - Transferring knowledge from larger to smaller models
- **Self-Play** - Improving models through iterative self-improvement

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- [Tinker API](https://thinkingmachines.ai/) access (for training infrastructure)
- Basic understanding of machine learning and LLMs

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PostTrain-Lab.git
cd PostTrain-Lab

# Install dependencies
pip install -r requirements.txt

# Set up your Tinker API key
export TINKER_API_KEY=<your-key-here>
```

### Run Your First Training

Start with supervised fine-tuning on the MBPP coding dataset:

```bash
# Open the SFT notebook
jupyter notebook methods/sft/sft_train.ipynb
```

This notebook will guide you through:
1. Loading and preparing the MBPP dataset
2. Setting up a LoRA training client with Llama-3.2-1B
3. Training the model for 50 steps
4. Comparing baseline vs. fine-tuned performance


## 📊 Results (Still a work in progress)

### SFT on MBPP (50 steps)

| Metric | Baseline | After SFT | Improvement |
|--------|----------|-----------|-------------|
| Loss   | 1.5811   | 0.0421    | 97.3% ↓     |
| Code Quality | ❌ Generic | ✅ Task-specific | Significant |

*Results from training on 200 examples, evaluated on held-out test set*


## 📁 Project Structure

```
PostTrain-Lab/
├── data/                    # Datasets for training and evaluation
│   ├── sft/                # Supervised fine-tuning data
│   ├── preference/         # Preference pairs for DPO/RLHF
│   ├── synthetic/          # Synthetically generated data
│   └── evaluation/         # Evaluation benchmarks
│
├── methods/                # Implementation of post-training techniques
│   ├── sft/               # ✅ Supervised Fine-Tuning
│   ├── dpo/               # 🚧 Direct Preference Optimization
│   ├── rlhf/              # 🚧 Reinforcement Learning from Human Feedback
│   ├── rlaif/             # 🚧 RL from AI Feedback
│   ├── rlvr/              # 🚧 RL with Verifiable Rewards
│   ├── continual/         # 🚧 Continual Learning
│   ├── distillation/      # 🚧 Knowledge Distillation
│   ├── selfplay/          # 🚧 Self-Play Training
│   └── finetuning/        # 🚧 Advanced Fine-Tuning Techniques
│
├── reward_models/          # Reward model implementations
├── trainers/               # Training utilities and loops
├── evaluators/             # Evaluation metrics and benchmarks
├── utils/                  # Shared utilities
└── scripts/                # Automation scripts
```

**Legend:** ✅ Complete | 🚧 In Progress | 📋 Planned


## 🛠️ Built With

- **[Tinker API](https://thinkingmachines.ai/)** - High-performance LLM training infrastructure
- **[Transformers](https://huggingface.co/transformers/)** - Tokenizers and model utilities
- **[Datasets](https://huggingface.co/docs/datasets/)** - Dataset loading and processing
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing

## 📚 Resources

### Papers & References
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [RLHF: Learning to Summarize](https://arxiv.org/abs/2009.01325)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)

### Tutorials
- [Tinker Documentation](https://thinkingmachines.ai/docs)
- [Understanding LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [RLHF Explained](https://huggingface.co/blog/rlhf)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Thinking Machines](https://thinkingmachines.ai/) for the 5k Grant.
- The open-source ML community for inspiration and tools
- All contributors who help make this project better

## 📬 Contact

Have questions or suggestions? Feel free to:
- Open an issue
- Start a discussion
- Reach out on Twitter/X

---

**⭐ Star this repo if you find it helpful!**

Built with ❤️ for the ML community
