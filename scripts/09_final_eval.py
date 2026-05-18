"""Stage 9: final internal evaluation (LM loss + throughput + lm-eval-harness).

Reference: plan section 11.
"""

from __future__ import annotations

from _common import base_parser, print_resolved

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

    # TODO(stage-9):
    #   Compare these checkpoints internally:
    #       - parent Nemotron-H 8B
    #       - best pruned candidate (no KD)
    #       - best pruned candidate after short KD
    #       - optional mini-final-KD model
    #   For each: measure_lm_loss, measure_throughput, run_harness.
    #   Write a CSV under base.paths.eval_dir / "09_final_eval.csv".
    raise NotImplementedError("TODO(stage-9): wire up final evaluation suite")


if __name__ == "__main__":
    raise SystemExit(main())
