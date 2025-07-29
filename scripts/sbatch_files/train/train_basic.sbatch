#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --output=OUTPUT_FILE_DIR
#SBATCH --error=ERROR_FILE_DIR
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:2
#SBATCH -c 4
#SBATCH --chdir=../..

# Load necessary modules (if needed)
export COMET_API_KEY=""
export COMET_workspace=""
export HF_HOME=""
export GOOGLE_API_KEY=""

conda activate llm_env
cd training

CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model Qwen/Qwen2.5-3B-Instruct --tensor_parallel_size 1 &

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29503 --config_file configs/deepspeed_zero3.yaml train.py --config configs/emnlp_experiments/basic.yaml

echo "All Python scripts finished."
