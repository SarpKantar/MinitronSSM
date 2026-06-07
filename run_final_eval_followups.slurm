#!/bin/bash
#SBATCH -J minitron_eval_followup
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --exclude=palamut3,palamut5,palamut6
#SBATCH --requeue
#SBATCH -o /arf/home/skantar/minitron_job_logs/final_eval/slurm-%A_%a.out
#SBATCH -e /arf/home/skantar/minitron_job_logs/final_eval/slurm-%A_%a.err

set -euo pipefail

REPO=/arf/scratch/skantar/MinitronSSM
cd "${REPO}"
mkdir -p logs/final_eval

source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron

STAMP=$(date +%Y%m%d-%H%M%S)
HOME_LOG_DIR=/arf/home/skantar/minitron_job_logs/final_eval
ln -sf "${HOME_LOG_DIR}/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
  "logs/final_eval/${STAMP}-followup-job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.out" \
  2>/dev/null || true
ln -sf "${HOME_LOG_DIR}/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" \
  "logs/final_eval/${STAMP}-followup-job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.err" \
  2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "[$(date)] Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started on $(hostname)"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

gpu_ready () {
  python - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print(
        f"expected exactly one visible CUDA GPU, got {torch.cuda.device_count()}",
        file=sys.stderr,
    )
    raise SystemExit(2)

free, total = torch.cuda.mem_get_info(0)
free_gib = free / 1024**3
total_gib = total / 1024**3
print(f"visible GPU preflight: free={free_gib:.2f} GiB total={total_gib:.2f} GiB")

# The student occupies about 10 GiB and the safe Mamba torch path can request a
# temporary 9 GiB tensor. Keep ample headroom so a foreign process cannot cause
# another late harness OOM.
raise SystemExit(0 if free_gib >= 48.0 else 3)
PY
}

if ! gpu_ready; then
  restart_count=${SLURM_RESTART_COUNT:-0}
  if (( restart_count < 5 )); then
    echo "[$(date)] Allocated GPU is contaminated; requeueing (attempt ${restart_count}/5)." >&2
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  fi
  echo "[$(date)] GPU preflight failed after ${restart_count} requeues." >&2
  exit 1
fi

STAGE_DIR="${TMPDIR:-/tmp}/minitron_eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${STAGE_DIR}"

# Direct sbatch uses the repo artifacts. The submit wrapper replaces this block
# with node-local JSON payloads embedded on the login node.
# __JSON_PAYLOAD_START__
KD_JSON_SOURCE="${REPO}/outputs/eval/08_kd_results_round2.json"
MINI_JSON_SOURCE="${REPO}/outputs/eval/10_mini_kd.json"
# __JSON_PAYLOAD_END__

stage_json () {
  local src=$1
  local dest=$2
  local attempt
  for attempt in $(seq 1 8); do
    if cp "${src}" "${dest}.tmp" 2>/dev/null \
      && python -m json.tool "${dest}.tmp" >/dev/null 2>&1; then
      mv "${dest}.tmp" "${dest}"
      return 0
    fi
    rm -f "${dest}.tmp"
    sleep "${attempt}"
  done
  echo "[$(date)] Failed to stage valid JSON: ${src}" >&2
  return 1
}

run_stage () {
  echo "[$(date)] >>> Running: $*"
  "$@"
  echo "[$(date)] <<< Finished: $*"
}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    stage_json \
      "${KD_JSON_SOURCE}" \
      "${STAGE_DIR}/08_kd_results_round2.json"
    run_stage python scripts/09b_final_eval_from_results.py \
      --kd-results "${STAGE_DIR}/08_kd_results_round2.json" \
      --output "${REPO}/outputs/eval/09_final_eval_round2.json" \
      --skip-parent \
      --skip-pruned \
      --harness-batch-size 1
    ;;
  1)
    stage_json \
      "${MINI_JSON_SOURCE}" \
      "${STAGE_DIR}/10_mini_kd.json"
    run_stage python scripts/09b_final_eval_from_results.py \
      --mini-kd-json "${STAGE_DIR}/10_mini_kd.json" \
      --output "${REPO}/outputs/eval/09_final_eval_mini_kd.json" \
      --skip-parent \
      --skip-pruned \
      --skip-kd \
      --harness-batch-size 1
    ;;
  *)
    echo "unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

echo "[$(date)] Job finished successfully"
