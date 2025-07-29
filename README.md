# 🧾 Judging by the Rules: Reward-Aligned Reasoning for

### Modern Slavery Statement Compliance Monitoring

Modern slavery remains a pervasive global issue, and NLP offers substantial potential to support regulatory compliance efforts. However, high-stakes compliance tasks require more than accurate classification—they demand transparent, interpretable reasoning to foster trust and ensure accountability. While recent work has applied large language models (LLMs) to compliance classification, these approaches often reduce complex regulatory assessments to binary decisions, lacking the structured reasoning needed for robust legal scrutiny.

We argue that compliance verification is inherently a reasoning problem: it involves evaluating whether textual statements adhere to well-defined regulatory rules. In this work, we propose a novel framework for aligning model reasoning with domain-specific compliance criteria in the context of modern slavery reporting.

At the core of our approach is the **Compliance Alignment Judge (CA-Judge)**, which evaluates model-generated justifications based on their fidelity to legal requirements. Building on this, we introduce **Compliance Alignment LLM (CALLM)**—a model fine-tuned to produce rule-consistent, interpretable reasoning. CALLM achieves improved predictive performance while delivering outputs that are both transparent and aligned with legal standards, offering a more reliable and adoptable solution for real-world compliance analysis.

---

## 📦 Code Overview

This repository includes:

* 🧠 Training code for **CALLM** using reward-aligned learning with CA-Judge
* ⚖️ Evaluation scripts for **CA-Judge** to assess reasoning quality

---
```bash
├── callm/
│   ├── scripts/                     # Scripts to launch training
│   │   ├── sbatch_files/           # SLURM batch scripts
│   │   │   └── sbatch files
│   │
│   ├── training/                   # Core model training and evaluation
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── model_utils.py
│   │   └── ...
├── README.md
```

## ⚙️ Setup Instructions for vLLM Environment

This guide walks you through setting up a Python environment using Miniconda for running experiments with [vLLM](https://github.com/vllm-project/vllm).

### 1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 2. Initialize Conda Environment

```bash
source ~/.bashrc
conda update -n base -c defaults conda
conda create -n llm_env python=3.10 -y
conda activate llm_env
```

### 3. Install Required Packages

```bash
# CUDA and PyTorch
conda install -c conda-forge cuda-nvcc=12.1 cudatoolkit=12.1 -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core libraries
pip install vllm
pip install trl accelerate deepspeed

# Utility libraries
pip install scikit-learn pynvml ipywidgets wandb comet_ml
```

---

## 🚀 Running CALLM and CA-Judge

### 🧠 Train CALLM with Reward-Aligned Supervision

Use SLURM to submit a training job:

```bash
cd scripts/sbatch_files/train
# Modify the sbatch file as needed
sbatch train_basic.sbatch # or sbatch train_judge.sbatch
```

### ⚖️ Run Evaluation

Evaluate model performance for a specific criterion:

```bash
cd scripts/sbatch_files/eval
# Modify the sbatch file as needed
sbatch eval_basic.sbatch # or sbatch eval_judge.sbatch
```

For questions, please contact the authors.
