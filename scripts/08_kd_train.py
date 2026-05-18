"""Stage 8: short KD on top-3 candidates (~200M tokens each).

Reference: plan section 9.2.
"""

from __future__ import annotations

from _common import base_parser, print_resolved

from minitron_ssm.utils.config import load_base, load_data, load_kd
from minitron_ssm.utils.logging import get_logger

log = get_logger("kd-train")


def main() -> int:
    p = base_parser(__doc__ or "")
    args = p.parse_args()

    base = load_base(args.base_config)
    data = load_data()
    kd = load_kd()

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("kd", kd)
        return 0

    # TODO(stage-8):
    #   For each of the top-3 candidates from 06_top3.json:
    #       trainer = KDTrainer(student, teacher, kd, ...)
    #       trainer.train(target_tokens=kd.budget.tokens_per_candidate)
    #       save resulting student under checkpoints_dir/<id>-kd/
    #       record final LM loss curve to base.paths.eval_dir
    raise NotImplementedError("TODO(stage-8): wire up top-3 KD training")


if __name__ == "__main__":
    raise SystemExit(main())
