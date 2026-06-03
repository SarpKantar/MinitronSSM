#!/bin/bash
#SBATCH -J minitron_kd_train
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o logs/kd_train/slurm-%j.out
#SBATCH -e logs/kd_train/slurm-%j.err

set -euo pipefail

echo "[$(date)] Job started on $(hostname)"

source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron
cd /arf/scratch/skantar/MinitronSSM

# Reduce CUDA allocator fragmentation; the Mamba torch_forward fallback allocates
# large short-lived fp32 tensors and benefits from expandable segments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage () {
  echo "[$(date)] >>> Running: $*"
  "$@"
  echo "[$(date)] <<< Finished: $*"
}

# Prerequisites: stages 03-06 must be complete (artifacts produced by
# run_overnight.slurm job 1280919 are already on disk).
# Stage 07 smoke test must also be complete (run_kd_smoke.slurm).
#
# Token budget per candidate: 20M (reduced from 200M).
# With online KD (teacher 8B + student ~4B) and torch_forward fallback at
# seq_len=8192 on one A100, expect ~6-8h per candidate = ~18-22h for 3.
# This fits inside the 24h hard limit with a ~2-6h safety margin.
#
# Stage 09 (final eval / lm-eval-harness) is NOT included here because
# it loads each model plus runs 7 benchmark tasks, which would easily
# push the total past 24h on top of stage 08.
run_stage python scripts/08_kd_train.py

echo "[$(date)] Job finished successfully"
