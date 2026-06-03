#!/bin/bash
#SBATCH -J minitron_overnight
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o logs/overnight/slurm-%j.out
#SBATCH -e logs/overnight/slurm-%j.err

set -euo pipefail

echo "[$(date)] Job started on $(hostname)"

# Conda
source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron

# Project
cd /arf/scratch/skantar/MinitronSSM

run_stage () {
  echo "[$(date)] >>> Running: $*"
  "$@"
  echo "[$(date)] <<< Finished: $*"
}

# You already completed stage 02, so continue from 03
run_stage python scripts/03_generate_candidates.py
run_stage python scripts/04_prune_candidates.py
run_stage python scripts/05_eval_candidates.py
run_stage python scripts/06_select_top3.py
run_stage python scripts/07_kd_smoke.py

# WARNING: these are very long; keep only if you really want full overnight run
# run_stage python scripts/08_kd_train.py
# run_stage python scripts/09_final_eval.py
# run_stage python scripts/10_optional_mini_kd.py

echo "[$(date)] Job finished successfully"