#!/bin/bash
# Submit through submit_paper_table4_eval.sh so EVAL_TARGET and paths are set.
#SBATCH -J minitron_table4
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=palamut3,palamut5,palamut6
#SBATCH --requeue
#SBATCH -o /arf/home/skantar/minitron_job_logs/table4/%x-%j.out
#SBATCH -e /arf/home/skantar/minitron_job_logs/table4/%x-%j.err

set -euo pipefail

REPO=/arf/scratch/skantar/MinitronSSM
cd "${REPO}"
mkdir -p logs/table4 /arf/home/skantar/minitron_job_logs/table4

source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron

: "${EVAL_TARGET:?submitter must set EVAL_TARGET}"
: "${EVAL_LABEL:?submitter must set EVAL_LABEL}"
: "${EVAL_OUTPUT:?submitter must set EVAL_OUTPUT}"

STAMP=$(date +%Y%m%d-%H%M%S)
HOME_LOG_DIR=/arf/home/skantar/minitron_job_logs/table4
ln -sf "${HOME_LOG_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
  "logs/table4/${STAMP}-${EVAL_LABEL}-job${SLURM_JOB_ID}.out" 2>/dev/null || true
ln -sf "${HOME_LOG_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" \
  "logs/table4/${STAMP}-${EVAL_LABEL}-job${SLURM_JOB_ID}.err" 2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

echo "[$(date)] Job ${SLURM_JOB_ID} target=${EVAL_TARGET} label=${EVAL_LABEL} on $(hostname)"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

set +e
python - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print(f"expected one visible CUDA GPU, got {torch.cuda.device_count()}", file=sys.stderr)
    raise SystemExit(2)
free, total = torch.cuda.mem_get_info(0)
free_gib = free / 1024**3
total_gib = total / 1024**3
print(f"GPU preflight: free={free_gib:.2f} GiB total={total_gib:.2f} GiB")
raise SystemExit(0 if free_gib >= 48.0 else 3)
PY
preflight_status=$?
set -e
if (( preflight_status != 0 )); then
  restart_count=${SLURM_RESTART_COUNT:-0}
  if (( preflight_status == 3 && restart_count < 5 )); then
    echo "[$(date)] GPU lacks headroom; requeueing attempt ${restart_count}/5" >&2
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  fi
  exit "${preflight_status}"
fi

ARGS=(
  --target "${EVAL_TARGET}"
  --label "${EVAL_LABEL}"
  --output "${EVAL_OUTPUT}"
  --batch-size 1
)
if [[ "${EVAL_TARGET}" == "checkpoint" ]]; then
  : "${EVAL_CHECKPOINT:?checkpoint target requires EVAL_CHECKPOINT}"
  ARGS+=(--checkpoint "${EVAL_CHECKPOINT}")
fi

echo "[$(date)] Starting paper-aligned Table 4 evaluation"
python scripts/11_paper_table4_eval.py "${ARGS[@]}"
echo "[$(date)] Job finished successfully"
