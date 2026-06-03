#!/bin/bash
#SBATCH -J minitron_kd_smoke
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o logs/kd_smoke/slurm-%j.out
#SBATCH -e logs/kd_smoke/slurm-%j.err

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

# Resume from existing artifacts produced by stages 03-06:
#   outputs/checkpoints/*
#   outputs/eval/05_candidates.json
#   outputs/eval/06_top3.json
#
# Keep this job to stage 07 only. Stage 08 trains top-3 candidates for
# 200M tokens each and is not a safe fit for a single 24h slot on one A100.
# Stage 09 also depends on stage 08 outputs for the KD model comparison.
run_stage python scripts/07_kd_smoke.py

echo "[$(date)] Job finished successfully"
