"""Stage 4: apply pruning to the parent for each candidate and checkpoint.

Reference: plan section 7.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from _common import base_parser, print_resolved

from minitron_ssm.models.load import load_parent
from minitron_ssm.pruning.apply import apply_candidate
from minitron_ssm.search.space import Candidate
from minitron_ssm.utils.checkpoint import save_candidate
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

    import torch

    parent, _ = load_parent(base, eval_mode=True)
    scores = torch.load(imp.output_path, map_location="cpu")

    # Stage 4 is pure tensor slicing – no forward passes required.
    # Move the parent to CPU so deepcopy inside apply_candidate does not
    # try to duplicate 16 GB of BF16 weights on the GPU (which exhausts
    # an 80 GB A100 by candidate 5–6).
    log.info("moving parent to CPU for memory-safe pruning ...")
    parent = parent.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidates_path = Path(base.paths.candidates_dir) / "candidates.json"
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"missing {candidates_path}; run scripts/03_generate_candidates.py first"
        )
    records = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = [Candidate(**r) for r in records]

    out_root = Path(base.paths.checkpoints_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    written = []
    for cand in candidates:
        log.info("pruning %s ...", cand.id)
        pruned = apply_candidate(parent, scores, cand)
        save_path = save_candidate(pruned.state_dict(), asdict(cand), out_root, cand.id)
        written.append({"candidate_id": cand.id, "path": str(save_path)})
        # Free the pruned copy immediately; the parent stays on CPU.
        del pruned
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = out_root / "04_pruned_manifest.json"
    manifest.write_text(json.dumps(written, indent=2), encoding="utf-8")
    log.info("wrote %s", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
