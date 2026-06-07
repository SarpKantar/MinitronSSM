#!/bin/bash
# Submit stage 10b with its stage-10 metadata embedded on the login node.
set -euo pipefail

REPO=/arf/scratch/skantar/MinitronSSM
cd "${REPO}"

TEMPLATE="${REPO}/run_mini_kd_round2.slurm"
SOURCE="${REPO}/outputs/eval/10_mini_kd.json"
python -m json.tool "${SOURCE}" >/dev/null

JOB_SCRIPT=$(mktemp /tmp/minitron_mini_kd_round2.XXXXXX.slurm)
trap 'rm -f "${JOB_SCRIPT}"' EXIT

python - "${TEMPLATE}" "${JOB_SCRIPT}" "${SOURCE}" <<'PY'
import sys
from pathlib import Path

template_path, job_path, source_path = sys.argv[1:4]
template = Path(template_path).read_text(encoding="utf-8")
start = "# __JSON_PAYLOAD_START__"
end = "# __JSON_PAYLOAD_END__"
before, rest = template.split(start, 1)
_, after = rest.split(end, 1)

source_json = Path(source_path).read_text(encoding="utf-8").rstrip()
payload = f"""{start}
PAYLOAD_DIR="${{TMPDIR:-/tmp}}/minitron_mini_kd_payload_${{SLURM_JOB_ID}}"
mkdir -p "${{PAYLOAD_DIR}}"
SOURCE_JSON_INPUT="${{PAYLOAD_DIR}}/10_mini_kd.json"
cat > "${{SOURCE_JSON_INPUT}}" <<'EOF_MINI_KD_SOURCE'
{source_json}
EOF_MINI_KD_SOURCE
{end}"""
Path(job_path).write_text(before + payload + after, encoding="utf-8")
PY

chmod +x "${JOB_SCRIPT}"
sbatch "${JOB_SCRIPT}"
