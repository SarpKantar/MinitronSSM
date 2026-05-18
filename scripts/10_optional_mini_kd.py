"""Stage 10 (optional): mini final KD on the single best candidate.

Reference: plan section 10.

This is the "if time remains" stage. Not equivalent to the paper's
380B-token final KD.
"""

from __future__ import annotations

from _common import base_parser, print_resolved

from minitron_ssm.utils.config import load_base, load_data, load_kd
from minitron_ssm.utils.logging import get_logger

log = get_logger("mini-final-kd")


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument(
        "--target-tokens",
        type=int,
        default=1_000_000_000,
        help="Additional KD tokens for the best candidate (200M to 1B).",
    )
    args = p.parse_args()

    base = load_base(args.base_config)
    data = load_data()
    kd = load_kd()

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("kd", kd)
        log.info("would train best candidate for %d additional tokens", args.target_tokens)
        return 0

    # TODO(stage-10):
    #   Pick the best post-KD candidate (lowest LM loss from 08).
    #   Continue KDTrainer.train(target_tokens=args.target_tokens).
    #   Save final checkpoint to checkpoints_dir/<id>-mini-final/.
    raise NotImplementedError("TODO(stage-10): wire up optional mini final KD")


if __name__ == "__main__":
    raise SystemExit(main())
