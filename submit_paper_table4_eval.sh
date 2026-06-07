#!/bin/bash
# Submit the paper-aligned Table 4 campaign while using at most three extra GPUs.
set -euo pipefail

REPO=/arf/scratch/skantar/MinitronSSM
cd "${REPO}"

TEMPLATE="${REPO}/run_paper_table4_eval.slurm"
KD_JOB_ID=${KD_JOB_ID:-1287432}

submit_eval () {
  local job_name=$1
  local target=$2
  local label=$3
  local output=$4
  local checkpoint=$5
  local dependency=${6:-}
  local export_vars="ALL,EVAL_TARGET=${target},EVAL_LABEL=${label},EVAL_OUTPUT=${output}"
  if [[ -n "${checkpoint}" ]]; then
    export_vars+=",EVAL_CHECKPOINT=${checkpoint}"
  fi

  local args=(
    --parsable
    --job-name "${job_name}"
    --export "${export_vars}"
  )
  if [[ -n "${dependency}" ]]; then
    args+=(--dependency "${dependency}")
  fi
  local submitted
  submitted=$(sbatch "${args[@]}" "${TEMPLATE}")
  printf '%s\n' "${submitted%%;*}"
}

PARENT_JOB=$(submit_eval \
  table4-parent \
  parent \
  parent-8b \
  "${REPO}/outputs/eval/11_table4_parent.json" \
  "")
echo "parent=${PARENT_JOB}"

REFERENCE_JOB=$(submit_eval \
  table4-reference \
  reference \
  official-4b \
  "${REPO}/outputs/eval/11_table4_reference4b.json" \
  "")
echo "reference=${REFERENCE_JOB}"

# Chain pre-KD behind the parent so the campaign starts with two GPUs and never
# exceeds three extra GPUs when the final KD checkpoint becomes available.
PRUNED_JOB=$(submit_eval \
  table4-pruned \
  checkpoint \
  cand-016-pre-kd \
  "${REPO}/outputs/eval/11_table4_pruned_cand016.json" \
  "${REPO}/outputs/checkpoints/cand-016" \
  "afterany:${PARENT_JOB}")
echo "pruned=${PRUNED_JOB}"

FINAL_JOB=$(submit_eval \
  table4-final \
  checkpoint \
  cand-016-mini-final2 \
  "${REPO}/outputs/eval/11_table4_final.json" \
  "${REPO}/outputs/checkpoints/cand-016-mini-final2" \
  "afterok:${KD_JOB_ID}")
echo "final=${FINAL_JOB}"

printf 'submitted parent=%s reference=%s pruned=%s final=%s\n' \
  "${PARENT_JOB}" "${REFERENCE_JOB}" "${PRUNED_JOB}" "${FINAL_JOB}"
