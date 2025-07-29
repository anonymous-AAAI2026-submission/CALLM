#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --output=OUTPUT_FILE_DIR
#SBATCH --error=ERROR_FILE_DIR
#SBATCH --chdir=../..

cd training
conda activate compliance_verifier
python run_compliance_verifier.py --input_csv INPUT_DIR --output_csv INPUT_DIR/compliance.csv --class CLASS_NAME