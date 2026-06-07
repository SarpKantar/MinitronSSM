#!/bin/bash
#SBATCH -J minitron_final_eval_r2
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

# Time-sortable, human-readable aliases for this job's logs.
LOG_DIR=logs/final_eval
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d-%H%M%S)
ln -sf "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/${STAMP}-round2-job${SLURM_JOB_ID}.out"
ln -sf "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/${STAMP}-round2-job${SLURM_JOB_ID}.err"

# The pruned/KD students run the Mamba torch_forward fallback during eval too;
# expandable segments keeps the allocator from fragmenting on transient tensors.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage () {
  echo "[$(date)] >>> Running: $*"
  "$@"
  echo "[$(date)] <<< Finished: $*"
}

# Stage 09b: final internal evaluation using the round-2 KD artifact.
# Prerequisites:
#   outputs/eval/06_top3.json
#   outputs/eval/08_kd_results_round2.json
#
# This evaluates parent, best_pruned, and the selected round-2 KD checkpoint.
# By default, 09b selects the KD record with the lowest last_loss and writes:
#   outputs/eval/09_final_eval_round2.json
#
# Add --kd-selection first to mirror scripts/09_final_eval.py's original
# "take row 0" behavior, or --skip-parent / --skip-pruned for a faster KD-only
# follow-up once comparable parent/pruned evals already exist.
# Submit with ./submit_final_eval_round2.sh so metadata JSON is embedded into the
# batch script on the login node. Direct sbatch of this template fails when
# /arf/scratch Lustre endpoints flap on compute nodes.
# __JSON_PAYLOAD_START__
echo "[$(date)] ERROR: submit via ./submit_final_eval_round2.sh, not sbatch on this template." >&2
exit 1
# __JSON_PAYLOAD_END__
EVAL_OUTPUT="${PWD}/outputs/eval/09_final_eval_round2.json"
# Parent/pruned harness scores already live in outputs/eval/09_final_eval.json.
# Skip them here to avoid lm-eval auto batch-size probing OOM on the 8B parent
# (Triton Mamba autotune on palamut3). Evaluate only the round-2 KD student
# with a fixed batch size; round-1 students used auto batch size 3.
run_stage python scripts/09b_final_eval_from_results.py \
  --kd-results "${KD_RESULTS}" \
  --top-results "${TOP_RESULTS}" \
  --output "${EVAL_OUTPUT}" \
  --skip-parent \
  --skip-pruned \
  --harness-batch-size 1

echo "[$(date)] Job finished successfully"
