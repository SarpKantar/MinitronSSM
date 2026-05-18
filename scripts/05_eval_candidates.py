"""Stage 5: zero-shot LM loss + throughput for every pruned candidate.

Reference: plan section 8.3.
"""

from __future__ import annotations

from _common import base_parser, print_resolved

from minitron_ssm.utils.config import load_base, load_data, load_eval
from minitron_ssm.utils.logging import get_logger

log = get_logger("eval-candidates")


def main() -> int:
    p = base_parser(__doc__ or "")
    args = p.parse_args()

    base = load_base(args.base_config)
    data = load_data()
    ev = load_eval()

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("eval", ev)
        return 0

    # TODO(stage-5):
    #   for each candidate ckpt under base.paths.checkpoints_dir:
    #       model, cfg = load_candidate(...)
    #       lm = measure_lm_loss(model, stream_validation(data), ...)
    #       tp = measure_throughput(model, tokenizer, ...)
    #       append result to base.paths.eval_dir / "05_candidates.json"
    raise NotImplementedError("TODO(stage-5): wire up candidate evaluation loop")


if __name__ == "__main__":
    raise SystemExit(main())
