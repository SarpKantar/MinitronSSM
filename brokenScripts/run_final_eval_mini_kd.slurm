#!/bin/bash
#SBATCH -J minitron_final_eval_mini
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

LOG_DIR=logs/final_eval
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d-%H%M%S)
ln -sf "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/${STAMP}-mini-job${SLURM_JOB_ID}.out"
ln -sf "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/${STAMP}-mini-job${SLURM_JOB_ID}.err"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage () {
  echo "[$(date)] >>> Running: $*"
  "$@"
  echo "[$(date)] <<< Finished: $*"
}

# Stage 09b (mini-final only): lm-eval-harness on the stage-10 checkpoint.
# Prerequisites: outputs/eval/10_mini_kd.json (from run_mini_kd.slurm).
# Parent/pruned/round-1/round-2 scores already exist in other 09* JSON files;
# this job evaluates only the extra 10M-token mini-final KD model.
#
# Submit with ./submit_final_eval_mini_kd.sh (embeds JSON on the login node).
# __JSON_PAYLOAD_START__
echo "[$(date)] ERROR: submit via ./submit_final_eval_mini_kd.sh, not sbatch on this template." >&2
exit 1
# __JSON_PAYLOAD_END__
EVAL_OUTPUT="${PWD}/outputs/eval/09_final_eval_mini_kd.json"
run_stage python scripts/09b_final_eval_from_results.py \
  --mini-kd-json "${MINI_KD_JSON}" \
  --output "${EVAL_OUTPUT}" \
  --skip-parent \
  --skip-pruned \
  --skip-kd \
  --harness-batch-size 1

echo "[$(date)] Job finished successfully"
