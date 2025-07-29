#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --output=OUTPUT_FILE_DIR
#SBATCH --error=ERROR_FILE_DIR
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100l:3
#SBATCH -c 24
#SBATCH --chdir=../..

# Load necessary modules (if needed)
export COMET_API_KEY=""
export COMET_workspace=""
export HF_HOME=""
export GOOGLE_API_KEY=""

conda activate llm_env
cd training

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

CUDA_VISIBLE_DEVICES=2 trl vllm-serve --model Qwen/Qwen2.5-3B-Instruct --tensor_parallel_size 1 &

CUDA_VISIBLE_DEVICES=1 python llm_judge_server.py --config configs/llmjudge_config.yaml --port 8009 &

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29503 --config_file configs/deepspeed_zero3.yaml train.py --config configs/emnlp_experiments/judge.yaml

echo "All Python scripts finished."
