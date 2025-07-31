#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --output=OUTPUT_FILE_DIR
#SBATCH --error=ERROR_FILE_DIR
#SBATCH --chdir=../..

# Load necessary modules (if needed)
export COMET_API_KEY=""
export COMET_workspace=""
export HF_HOME=""
export GOOGLE_API_KEY=""

conda activate llm_env
cd training

python test.py --config configs/aaai_evals/basic.yaml

echo "All Python scripts finished."
