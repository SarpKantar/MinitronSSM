"""Stage 9: final internal evaluation (LM loss + throughput + lm-eval-harness).

Reference: plan section 11.
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import base_parser, print_resolved

from minitron_ssm.eval.harness import run_harness
from minitron_ssm.utils.config import load_base, load_eval
from minitron_ssm.utils.logging import get_logger

log = get_logger("final-eval")


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument(
        "--include-optional",
        action="store_true",
        help="Also run GSM8K / HumanEval / MBPP if time allows.",
    )
    args = p.parse_args()

    base = load_base(args.base_config)
    ev = load_eval()

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("eval", ev)
        return 0

    eval_dir = Path(base.paths.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    top_path = eval_dir / "06_top3.json"
    kd_path = eval_dir / "08_kd_results.json"
    models = [
        {"name": "parent", "checkpoint": base.models.parent},
    ]
    if top_path.exists():
        top = json.loads(top_path.read_text(encoding="utf-8"))
        if top:
            models.append({"name": "best_pruned", "checkpoint": top[0]["checkpoint"]})
    if kd_path.exists():
        kd = json.loads(kd_path.read_text(encoding="utf-8"))
        if kd:
            models.append({"name": "best_kd", "checkpoint": kd[0]["checkpoint"]})

    results = []
    for entry in models:
        harness = run_harness(
            entry["checkpoint"],
            ev,
            include_optional=args.include_optional,
        )
        results.append(
            {
                "model": entry["name"],
                "checkpoint": entry["checkpoint"],
                "harness": harness,
            }
        )
        log.info("evaluated %s", entry["name"])

    out_path = eval_dir / "09_final_eval.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
