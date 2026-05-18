"""Stage 3: enumerate ~20 candidate architectures.

Reference: plan section 8.

Fully implemented on CPU (no model required).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from _common import base_parser, print_resolved

from minitron_ssm.search.param_count import parent_arch_from_base, count_params
from minitron_ssm.search.space import enumerate_candidates
from minitron_ssm.utils.config import load_base, load_search_space
from minitron_ssm.utils.logging import get_logger

log = get_logger("candidates")


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write candidates.json (defaults to base.paths.candidates_dir).",
    )
    args = p.parse_args()

    base = load_base(args.base_config)
    space = load_search_space()
    parent = parent_arch_from_base(base)

    log.info("parent analytical param count: %.2fB", count_params(parent) / 1e9)

    candidates = enumerate_candidates(space, parent)
    log.info("emitted %d candidates", len(candidates))

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("search_space", space)
        for c in candidates:
            print(
                f"  {c.id}: layers={c.layers} emb={c.embedding} ffn={c.ffn} "
                f"heads={c.mamba_heads} hc={c.mamba_head_channels} "
                f"params={c.param_count/1e9:.2f}B"
            )
        return 0

    out_dir = args.output or Path(base.paths.candidates_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in candidates], f, indent=2)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
