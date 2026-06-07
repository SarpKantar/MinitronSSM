#!/bin/bash
# Active Slurm launcher; run from the repository root.
#SBATCH -J minitron_final_eval
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o logs/final_eval/slurm-%j.out
#SBATCH -e logs/final_eval/slurm-%j.err

set -euo pipefail

echo "[$(date)] Job started on $(hostname)"

source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron
cd /arf/scratch/skantar/MinitronSSM

# Time-sortable, human-readable aliases for this job's logs. Slurm writes the live
# logs to slurm-<jobid>.{out,err}; these symlinks prefix the job start time
# (YYYYMMDD-HHMMSS) so `ls logs/final_eval` shows oldest -> newest at a glance.
LOG_DIR=logs/final_eval
STAMP=$(date +%Y%m%d-%H%M%S)
ln -sf "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/${STAMP}-job${SLURM_JOB_ID}.out"
ln -sf "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/${STAMP}-job${SLURM_JOB_ID}.err"

# The pruned/KD students run the Mamba torch_forward fallback during eval too;
# expandable segments keeps the allocator from fragmenting on the transient tensors.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage () {
  echo "[$(date)] >>> Running: $*"
  "$@"
  echo "[$(date)] <<< Finished: $*"
}

# Stage 09: final internal evaluation (lm-eval-harness Table-4 task list).
# Prerequisites: stage 06 (06_top3.json) and stage 08 (08_kd_results.json).
#
# Evaluates three models in-memory, one at a time (parent stays resident as the
# architecture template; each student is built + freed in turn):
#   parent       -> base.models.parent (HF checkpoint)
#   best_pruned  -> 06_top3.json[0]   (pre-KD pruned candidate)
#   best_kd      -> 08_kd_results.json[0] (KD-recovered candidate)
# Tasks (configs/eval.yaml, num_fewshot=0): arc_challenge, arc_easy, hellaswag,
# piqa, winogrande, openbookqa, mmlu. Output: outputs/eval/09_final_eval.json.
#
# Add --include-optional to also run gsm8k / humaneval / mbpp (much slower; only
# if you have spare wall-clock).
run_stage python scripts/09_final_eval.py

echo "[$(date)] Job finished successfully"
