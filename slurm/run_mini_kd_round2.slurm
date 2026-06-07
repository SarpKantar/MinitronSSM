#!/bin/bash
# Active Slurm launcher; run from the repository root.
#SBATCH -J minitron_mini_kd_r2
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --exclude=palamut3,palamut5,palamut6
#SBATCH --requeue
#SBATCH -o /arf/home/skantar/minitron_job_logs/mini_kd/slurm-%j.out
#SBATCH -e /arf/home/skantar/minitron_job_logs/mini_kd/slurm-%j.err

set -euo pipefail

REPO=/arf/scratch/skantar/MinitronSSM
cd "${REPO}"
mkdir -p logs/mini_kd

source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron

STAMP=$(date +%Y%m%d-%H%M%S)
HOME_LOG_DIR=/arf/home/skantar/minitron_job_logs/mini_kd
ln -sf "${HOME_LOG_DIR}/slurm-${SLURM_JOB_ID}.out" \
  "logs/mini_kd/${STAMP}-round2-job${SLURM_JOB_ID}.out" \
  2>/dev/null || true
ln -sf "${HOME_LOG_DIR}/slurm-${SLURM_JOB_ID}.err" \
  "logs/mini_kd/${STAMP}-round2-job${SLURM_JOB_ID}.err" \
  2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "[$(date)] Job ${SLURM_JOB_ID} started on $(hostname)"
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

# Stage 10 previously fit on an otherwise empty A100. Require substantial
# headroom before loading teacher + student and requeue contaminated allocations.
raise SystemExit(0 if free_gib >= 64.0 else 3)
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

STAGE_DIR="${TMPDIR:-/tmp}/minitron_mini_kd_r2_${SLURM_JOB_ID}"
mkdir -p "${STAGE_DIR}"
SOURCE_JSON="${STAGE_DIR}/10_mini_kd.json"

# Direct sbatch uses the repo artifact. The submit wrapper replaces this block
# with a node-local JSON payload embedded on the login node.
# __JSON_PAYLOAD_START__
SOURCE_JSON_INPUT="${REPO}/outputs/eval/10_mini_kd.json"
# __JSON_PAYLOAD_END__

for attempt in $(seq 1 8); do
  if cp "${SOURCE_JSON_INPUT}" "${SOURCE_JSON}.tmp" 2>/dev/null \
    && python -m json.tool "${SOURCE_JSON}.tmp" >/dev/null 2>&1; then
    mv "${SOURCE_JSON}.tmp" "${SOURCE_JSON}"
    break
  fi
  rm -f "${SOURCE_JSON}.tmp"
  if (( attempt == 8 )); then
    echo "[$(date)] Failed to stage outputs/eval/10_mini_kd.json" >&2
    exit 1
  fi
  sleep "${attempt}"
done

echo "[$(date)] >>> Running stage 10b"
python scripts/10b_mini_kd_round2.py \
  --source "${SOURCE_JSON}" \
  --output "${REPO}/outputs/eval/10_mini_kd_round2.json" \
  --target-tokens 10_000_000
echo "[$(date)] <<< Finished stage 10b"

echo "[$(date)] Job finished successfully"
