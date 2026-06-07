#!/bin/bash
# Active Slurm launcher; run from the repository root.
#SBATCH -J minitron_kd_round2
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o logs/kd_train/slurm-%A_%a.out
#SBATCH -e logs/kd_train/slurm-%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

echo "[$(date)] Job started on $(hostname) (array job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID})"

source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron
cd /arf/scratch/skantar/MinitronSSM

# Time-sortable, human-readable aliases for this job's logs.
LOG_DIR=logs/kd_train
STAMP=$(date +%Y%m%d-%H%M%S)
BASE="slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
ln -sf "${BASE}.out" "${LOG_DIR}/${STAMP}-round2-job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.out"
ln -sf "${BASE}.err" "${LOG_DIR}/${STAMP}-round2-job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.err"

# Reduce CUDA allocator fragmentation; the Mamba torch_forward fallback allocates
# large short-lived fp32 tensors and benefits from expandable segments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage () {
  echo "[$(date)] >>> Running: $*"
  "$@"
  echo "[$(date)] <<< Finished: $*"
}

# Stage 08b: SECOND round of KD (extra training when spare compute remains).
# Resumes each top candidate from its stage-08 `-kd` checkpoint and trains
# another `tokens_per_candidate` (15M) tokens on a fresh data seed, taking each
# candidate to ~30M total tokens (~14.5% of the paper's 200M, up from 7.5%).
#
# One candidate per array task (0-2), each its own GPU + 24h window. Round 1 ran
# 15M in ~19h, so this fits with margin. Outputs:
#   outputs/checkpoints/cand-XXX-kd2/
#   outputs/eval/08_kd_results_round2.candNN.json  (+ merged 08_kd_results_round2.json)
# The original stage-08 artifacts are left untouched.
#
# Prerequisite: outputs/eval/08_kd_results.json must exist (stage 08 round 1 done).
run_stage python scripts/08b_kd_train_round2.py --candidate-index "${SLURM_ARRAY_TASK_ID}"

echo "[$(date)] Job finished successfully (task ${SLURM_ARRAY_TASK_ID})"
