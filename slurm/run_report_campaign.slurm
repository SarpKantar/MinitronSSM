#!/bin/bash
# Active Slurm launcher; run from the repository root.
# Submitted as a five-task array by slurm/submit_report_campaign.sh.
#SBATCH -J minitron_report
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --exclude=palamut3,palamut5,palamut6
#SBATCH --requeue
#SBATCH -o /arf/home/skantar/minitron_job_logs/report/slurm-%A_%a.out
#SBATCH -e /arf/home/skantar/minitron_job_logs/report/slurm-%A_%a.err

set -euo pipefail

REPO=/arf/scratch/skantar/MinitronSSM
cd "${REPO}"
mkdir -p logs/report /arf/home/skantar/minitron_job_logs/report

source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron

STAMP=$(date +%Y%m%d-%H%M%S)
HOME_LOG_DIR=/arf/home/skantar/minitron_job_logs/report
ln -sf "${HOME_LOG_DIR}/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
  "logs/report/${STAMP}-job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.out" \
  2>/dev/null || true
ln -sf "${HOME_LOG_DIR}/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" \
  "logs/report/${STAMP}-job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.err" \
  2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "[$(date)] Report task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
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
print(f"GPU preflight: free={free_gib:.2f} GiB total={total / 1024**3:.2f} GiB")
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

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python scripts/12_report_suite_eval.py \
      --target checkpoint \
      --checkpoint outputs/checkpoints/cand-016 \
      --label cand-016-pre-kd \
      --output outputs/eval/12_report_pre_kd.json \
      --batch-size 1
    ;;
  1)
    python scripts/12_report_suite_eval.py \
      --target checkpoint \
      --checkpoint outputs/checkpoints/cand-016-kd \
      --label cand-016-kd-15m \
      --output outputs/eval/12_report_kd15m.json \
      --batch-size 1
    ;;
  2)
    python scripts/12_report_suite_eval.py \
      --target checkpoint \
      --checkpoint outputs/checkpoints/cand-016-mini-final2 \
      --label cand-016-mini-final2-35m \
      --output outputs/eval/12_report_final35m.json \
      --batch-size 1
    ;;
  3)
    python scripts/12_report_suite_eval.py \
      --target reference \
      --label official-nemotron-h-4b-native \
      --output outputs/eval/12_report_reference4b_native.json \
      --batch-size 1
    ;;
  4)
    python scripts/12_report_efficiency.py \
      --checkpoint outputs/checkpoints/cand-016-mini-final2 \
      --label cand-016-mini-final2-35m \
      --output outputs/eval/12_report_final35m_efficiency.json \
      --num-loss-tokens 1000000
    ;;
  *)
    echo "unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

echo "[$(date)] Report task finished successfully"
