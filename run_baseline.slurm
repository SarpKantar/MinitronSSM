#!/bin/bash
#SBATCH -J minitron_baseline
#SBATCH -p palamut-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o logs/baseline/slurm-%j.out
#SBATCH -e logs/baseline/slurm-%j.err

set -euo pipefail
source /arf/home/skantar/anaconda3/etc/profile.d/conda.sh
conda activate minitron
cd /arf/scratch/skantar/MinitronSSM

# Time-sortable, human-readable aliases for this job's logs. Slurm writes the live
# logs to slurm-<jobid>.{out,err}; these symlinks prefix the job start time
# (YYYYMMDD-HHMMSS) so `ls logs/baseline` shows oldest -> newest at a glance.
LOG_DIR=logs/baseline
STAMP=$(date +%Y%m%d-%H%M%S)
ln -sf "slurm-${SLURM_JOB_ID}.out" "${LOG_DIR}/${STAMP}-job${SLURM_JOB_ID}.out"
ln -sf "slurm-${SLURM_JOB_ID}.err" "${LOG_DIR}/${STAMP}-job${SLURM_JOB_ID}.err"

python scripts/01_baseline.py