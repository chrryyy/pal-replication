#!/bin/bash
 
#SBATCH --job-name="pal"
#SBATCH --output=job%x-%j.out
#SBATCH --error=job_%x-%j.err
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpupod
#SBATCH --mem=512GB

export HF_HOME=/scratch/eqk9vb/.cache/huggingface/hub

# Print environment for debugging
echo "Using cache directory: $HF_HOME"

python3 gsm_eval.py