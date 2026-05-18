"""Stage 7: short KD smoke test (~5-20M tokens) before committing to full runs.

Reference: plan section 9.2.
"""

from __future__ import annotations

from _common import base_parser, print_resolved

from minitron_ssm.utils.config import load_base, load_data, load_kd
from minitron_ssm.utils.logging import get_logger

log = get_logger("kd-smoke")


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

    # TODO(stage-7):
    #   1. Load top-1 candidate from 06_top3.json.
    #   2. Build mixed_stream(data) -> packed_token_batches(...).
    #   3. teacher = load_parent(base); student = load_candidate(...).
    #   4. trainer = KDTrainer(student, teacher, kd, train_loader, val_loader).
    #   5. trainer.train(target_tokens=kd.budget.smoke_test_tokens).
    #   6. Sanity-check that loss decreases.
    raise NotImplementedError("TODO(stage-7): wire up KD smoke test")


if __name__ == "__main__":
    raise SystemExit(main())
