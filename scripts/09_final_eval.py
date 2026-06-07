"""Stage 9: final internal evaluation (LM loss + throughput + lm-eval-harness).

Reference: plan section 11.
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import base_parser, print_resolved

from minitron_ssm.eval.harness import run_harness
from minitron_ssm.models.load import (
    build_pruned_model,
    disable_fast_mamba_kernels,
    load_parent,
)
from minitron_ssm.utils.checkpoint import load_candidate
from minitron_ssm.utils.config import load_base, load_eval
from minitron_ssm.utils.logging import get_logger

log = get_logger("final-eval")


def _build_student(parent, checkpoint: Path):
    """Reconstruct a pruned/KD student from a candidate checkpoint.

    The on-disk ``config.json`` is candidate metadata (not an HF config), so we
    rebuild the shell from the parent config + candidate dims and load weights —
    the same path stages 07/08 use — instead of ``from_pretrained``.
    """
    state_dict, cand_cfg = load_candidate(checkpoint)
    student = build_pruned_model(parent, cand_cfg, eval_mode=True)
    student.load_state_dict(state_dict, strict=True)
    patched = disable_fast_mamba_kernels(student)
    if patched:
        log.info("disabled fused Mamba kernels for %d module(s)", patched)
    device = next(parent.parameters()).device
    return student.to(device).eval()


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

    import torch

    eval_dir = Path(base.paths.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    top_path = eval_dir / "06_top3.json"
    kd_path = eval_dir / "08_kd_results.json"

    # Parent loaded once; reused as the architecture template for the pruned/KD
    # shells and evaluated in-memory itself.
    parent, tokenizer = load_parent(base, eval_mode=True)

    plan = [{"name": "parent", "model": parent, "checkpoint": base.models.parent}]
    if top_path.exists():
        top = json.loads(top_path.read_text(encoding="utf-8"))
        if top:
            plan.append({"name": "best_pruned", "checkpoint": top[0]["checkpoint"]})
    if kd_path.exists():
        kd = json.loads(kd_path.read_text(encoding="utf-8"))
        if kd:
            plan.append({"name": "best_kd", "checkpoint": kd[0]["checkpoint"]})

    results = []
    for entry in plan:
        name = entry["name"]
        student = entry.get("model")
        built_here = student is None
        if built_here:
            log.info("building %s from %s", name, entry["checkpoint"])
            student = _build_student(parent, Path(entry["checkpoint"]))
        harness = run_harness(
            student,
            ev,
            include_optional=args.include_optional,
            tokenizer=tokenizer,
        )
        results.append(
            {
                "model": name,
                "checkpoint": str(entry["checkpoint"]),
                "harness": harness,
            }
        )
        log.info("evaluated %s", name)
        if built_here:
            del student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out_path = eval_dir / "09_final_eval.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
