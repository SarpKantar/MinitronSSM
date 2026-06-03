#!/bin/bash
#SBATCH -J minitron_baseline
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o logs/baseline/slurm-%j.out
#SBATCH -e logs/baseline/slurm-%j.err

set -euo pipefail
source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron
cd /arf/scratch/skantar/MinitronSSM

python scripts/01_baseline.py