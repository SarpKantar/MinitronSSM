"""Stage 6: rank candidates by (LM loss, throughput) and write top-3.

Reference: plan section 8.3.
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import base_parser

from minitron_ssm.utils.config import load_base
from minitron_ssm.utils.logging import get_logger

log = get_logger("select")


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    base = load_base(args.base_config)

    eval_path = Path(base.paths.eval_dir) / "05_candidates.json"
    if args.dry_run:
        log.info("would read %s and select top %d", eval_path, args.top_k)
        return 0

    if not eval_path.exists():
        raise FileNotFoundError(f"missing {eval_path}; run 05_eval_candidates first")

    with eval_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    # Lower LM loss first; tie-break by higher throughput.
    records.sort(key=lambda r: (r["lm_loss"], -r.get("tokens_per_second", 0.0)))
    top = records[: args.top_k]

    out_path = Path(base.paths.eval_dir) / "06_top3.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(top, f, indent=2)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
