"""Stage 4: apply pruning to the parent for each candidate and checkpoint.

Reference: plan section 7.
"""

from __future__ import annotations

from _common import base_parser, print_resolved

from minitron_ssm.utils.config import load_base, load_importance, load_search_space
from minitron_ssm.utils.logging import get_logger

log = get_logger("prune")


def main() -> int:
    p = base_parser(__doc__ or "")
    args = p.parse_args()

    base = load_base(args.base_config)
    space = load_search_space()
    imp = load_importance()

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("search_space", space)
        print_resolved("importance", imp)
        return 0

    # TODO(stage-4):
    #   1. parent, _ = load_parent(base)
    #   2. scores = torch.load(imp.output_path)
    #   3. for candidate in load(candidates.json):
    #        pruned = apply_candidate(parent, scores, candidate)
    #        assert_shapes(pruned.state_dict_shapes(), candidate_to_arch(candidate))
    #        save_candidate(pruned.state_dict(), asdict(candidate),
    #                       Path(base.paths.checkpoints_dir), candidate.id)
    raise NotImplementedError("TODO(stage-4): wire up per-candidate pruning")


if __name__ == "__main__":
    raise SystemExit(main())
