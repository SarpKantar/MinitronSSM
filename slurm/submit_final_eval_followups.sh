#!/bin/bash
# Active Slurm submission wrapper; run from the repository root.
# Submit the round-2 and mini-final harness evaluations with metadata embedded.
set -euo pipefail

REPO=/arf/scratch/skantar/MinitronSSM
cd "${REPO}"

TEMPLATE="${REPO}/slurm/run_final_eval_followups.slurm"
KD_SRC="${REPO}/outputs/eval/08_kd_results_round2.json"
MINI_SRC="${REPO}/outputs/eval/10_mini_kd.json"

for src in "${KD_SRC}" "${MINI_SRC}"; do
  python -m json.tool "${src}" >/dev/null
done

JOB_SCRIPT=$(mktemp /tmp/minitron_eval_followups.XXXXXX.slurm)
trap 'rm -f "${JOB_SCRIPT}"' EXIT

python - "${TEMPLATE}" "${JOB_SCRIPT}" "${KD_SRC}" "${MINI_SRC}" <<'PY'
import sys
from pathlib import Path

template_path, job_path, kd_path, mini_path = sys.argv[1:5]
template = Path(template_path).read_text(encoding="utf-8")
start = "# __JSON_PAYLOAD_START__"
end = "# __JSON_PAYLOAD_END__"
before, rest = template.split(start, 1)
_, after = rest.split(end, 1)

kd_json = Path(kd_path).read_text(encoding="utf-8").rstrip()
mini_json = Path(mini_path).read_text(encoding="utf-8").rstrip()
payload = f"""{start}
PAYLOAD_DIR="${{TMPDIR:-/tmp}}/minitron_eval_payload_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "${{PAYLOAD_DIR}}"
KD_JSON_SOURCE="${{PAYLOAD_DIR}}/08_kd_results_round2.json"
MINI_JSON_SOURCE="${{PAYLOAD_DIR}}/10_mini_kd.json"
cat > "${{KD_JSON_SOURCE}}" <<'EOF_KD_RESULTS'
{kd_json}
EOF_KD_RESULTS
cat > "${{MINI_JSON_SOURCE}}" <<'EOF_MINI_RESULTS'
{mini_json}
EOF_MINI_RESULTS
{end}"""
Path(job_path).write_text(before + payload + after, encoding="utf-8")
PY

chmod +x "${JOB_SCRIPT}"
sbatch "${JOB_SCRIPT}"
