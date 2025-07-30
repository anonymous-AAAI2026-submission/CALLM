# 🧾 Judging by the Rules: Compliance-Aligned Framework for \\ Modern Slavery Statement Monitoring


Modern slavery affects millions of people worldwide, and regulatory frameworks such as Modern Slavery Acts now require companies to publish detailed disclosures. However, these statements are often vague and inconsistent, making manual review very slow. NLP offers a promising path forward, but high-stakes compliance tasks require more than accurate classification. They demand transparent, rule-aligned outputs that legal experts can verify. Existing applications of large language models (LLMs)  often reduce complex regulatory assessments to binary decisions, lacking the structure needed for robust legal scrutiny.

We argue that compliance verification is fundamentally a rule-matching problem: it involves evaluating whether textual statements adhere to well-defined regulatory rules. To address this, we propose a novel framework for aligning model outputs with domain-specific compliance criteria.

At its core is the **Compliance Alignment Judge (CA-Judge)**, which evaluates model-generated justifications based on their fidelity to legal requirements.  Using this feedback, we train **Compliance Alignment LLM (CALLM)**, a model that produces rule-consistent, human-verifiable outputs. CALLM achieves improved predictive performance and generates outputs that are both transparent and aligned with legal standards, offering a more verifiable and actionable solution for real-world compliance analysis.

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
