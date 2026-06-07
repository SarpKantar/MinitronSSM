#!/bin/bash
#SBATCH -J minitron_mini_kd_r2
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

# Time-sortable, human-readable aliases for this job's logs.
LOG_DIR=logs/mini_kd
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d-%H%M%S)
ln -sf "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/${STAMP}-round2-job${SLURM_JOB_ID}.out"
ln -sf "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/${STAMP}-round2-job${SLURM_JOB_ID}.err"

# Reduce CUDA allocator fragmentation; the Mamba torch_forward fallback allocates
# large short-lived fp32 tensors and benefits from expandable segments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage () {
  echo "[$(date)] >>> Running: $*"
  "$@"
  echo "[$(date)] <<< Finished: $*"
}

# Stage 10b (OPTIONAL): continue from stage 10's mini-final checkpoint for one
# more 10M-token pass on a fresh data seed. This leaves 10_mini_kd.json and the
# first mini-final checkpoint intact. Outputs:
#   outputs/checkpoints/cand-XXX-mini-final2/
#   outputs/eval/10_mini_kd_round2.json
#
# Prerequisite: outputs/eval/10_mini_kd.json from run_mini_kd.slurm.
run_stage python scripts/10b_mini_kd_round2.py --target-tokens 10_000_000

echo "[$(date)] Job finished successfully"
