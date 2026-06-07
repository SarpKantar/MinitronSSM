#!/bin/bash
#SBATCH -J minitron_kd_train
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

# Time-sortable, human-readable aliases for this job's logs. Slurm writes the live
# logs to slurm-<arrayjob>_<task>.{out,err}; these symlinks prefix the job start
# time (YYYYMMDD-HHMMSS) so `ls logs/kd_train` shows oldest -> newest at a glance.
LOG_DIR=logs/kd_train
STAMP=$(date +%Y%m%d-%H%M%S)
BASE="slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
ln -sf "${BASE}.out" "${LOG_DIR}/${STAMP}-job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.out"
ln -sf "${BASE}.err" "${LOG_DIR}/${STAMP}-job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.err"

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
# ONE CANDIDATE PER JOB: this is a Slurm array (0-2) over the top-3 candidates
# in outputs/eval/06_top3.json. Each array task trains exactly one candidate and
# gets its own 24h slot + GPU, so a single candidate no longer has to share the
# wall-clock budget with the other two. Each task writes a per-candidate result
# (outputs/eval/08_kd_results.candNN.json) and re-merges the aggregate
# outputs/eval/08_kd_results.json (ordered by candidate index) that stage 09 reads.
#
# Token budget per candidate: 15M (configs/kd.yaml). At seq_len=1024 / ~4.7 s/step
# that is ~14.6k steps ≈ 19h, inside the 24h hard limit with margin.
#
# If the cluster has <3 free GPUs the array tasks queue and run as slots free up;
# either way every candidate gets a full, independent 24h window.
run_stage python scripts/08_kd_train.py --candidate-index "${SLURM_ARRAY_TASK_ID}"

echo "[$(date)] Job finished successfully (task ${SLURM_ARRAY_TASK_ID})"
