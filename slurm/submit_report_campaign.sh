#!/bin/bash
# Active Slurm submission wrapper; run from the repository root.
set -euo pipefail

REPO=/arf/scratch/skantar/MinitronSSM
cd "${REPO}"
mkdir -p logs/report /arf/home/skantar/minitron_job_logs/report

sbatch --parsable "${REPO}/slurm/run_report_campaign.slurm"
