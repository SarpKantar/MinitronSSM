#!/bin/bash
#SBATCH -J minitron_mini_kd
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o logs/mini_kd/slurm-%j.out
#SBATCH -e logs/mini_kd/slurm-%j.err

set -euo pipefail

echo "[$(date)] Job started on $(hostname)"

source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron
cd /arf/scratch/skantar/MinitronSSM

# Time-sortable, human-readable aliases for this job's logs. Slurm writes the live
# logs to slurm-<jobid>.{out,err}; these symlinks prefix the job start time
# (YYYYMMDD-HHMMSS) so `ls logs/mini_kd` shows oldest -> newest at a glance.
LOG_DIR=logs/mini_kd
STAMP=$(date +%Y%m%d-%H%M%S)
ln -sf "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/${STAMP}-job${SLURM_JOB_ID}.out"
ln -sf "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/${STAMP}-job${SLURM_JOB_ID}.err"

# Reduce CUDA allocator fragmentation; the Mamba torch_forward fallback allocates
# large short-lived fp32 tensors and benefits from expandable segments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage () {
  echo "[$(date)] >>> Running: $*"
  "$@"
  echo "[$(date)] <<< Finished: $*"
}

# Stage 10 (OPTIONAL): a "mini" final KD that continues training the single best
# KD candidate (lowest last_loss in 08_kd_results.json) for additional tokens.
# This is NOT the paper's 380B-token final KD -- it is an "if time remains" stage.
#
# The script default is --target-tokens 1_000_000_000 (1B), which is unrealistic
# on the torch_forward fallback (~4.7 s/step, 1024 tok/step => ~1.5 months). We
# pass a budget that fits a single 24h slot with a LARGE safety margin (so a bad
# per-step estimate can't lose the whole run):
#   10M tokens / 1024 = ~9.77k steps. At 4.7 s/step ≈ 12.8 h; even at a
#   pessimistic 5.5 s/step ≈ 14.9 h -- ~9-11 h of head-room under the 24h limit.
run_stage python scripts/10_optional_mini_kd.py --target-tokens 10_000_000

echo "[$(date)] Job finished successfully"
