"""Stage 10 (optional): mini final KD on the single best candidate.

Reference: plan section 10.

This is the "if time remains" stage. Not equivalent to the paper's
380B-token final KD.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from _common import base_parser, print_resolved

from minitron_ssm.data.mixture import mixed_stream
from minitron_ssm.data.tokenize import packed_token_batches
from minitron_ssm.kd.trainer import KDTrainer
from minitron_ssm.models.load import load_parent
from minitron_ssm.utils.checkpoint import load_candidate, save_candidate
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

    kd_path = Path(base.paths.eval_dir) / "08_kd_results.json"
    if not kd_path.exists():
        raise FileNotFoundError(f"missing {kd_path}; run scripts/08_kd_train.py first")
    kd_results = json.loads(kd_path.read_text(encoding="utf-8"))
    if not kd_results:
        raise ValueError("08_kd_results.json is empty")

    best = min(kd_results, key=lambda x: x.get("last_loss", float("inf")))
    teacher, tokenizer = load_parent(base, eval_mode=True)
    student = copy.deepcopy(teacher)
    state_dict, cand_cfg = load_candidate(Path(best["checkpoint"]))
    student.load_state_dict(state_dict, strict=False)
    student.train()

    stream = mixed_stream(data, seed=base.seed)
    train_loader = packed_token_batches(
        stream,
        tokenizer=tokenizer,
        seq_len=kd.training.seq_len,
        batch_size=max(1, int(kd.training.micro_batch_size)),
    )
    trainer = KDTrainer(student, teacher, kd, train_loader, val_loader=None)
    state = trainer.train(target_tokens=int(args.target_tokens))

    out_dir = Path(base.paths.checkpoints_dir)
    save_path = save_candidate(
        student.state_dict(),
        cand_cfg,
        out_dir,
        f"{cand_cfg.get('id', best['candidate_id'])}-mini-final",
    )
    out_path = Path(base.paths.eval_dir) / "10_mini_kd.json"
    payload = {
        "candidate_id": cand_cfg.get("id", best["candidate_id"]),
        "checkpoint": str(save_path),
        "tokens_seen": state.tokens_seen,
        "steps": state.step,
        "last_loss": state.last_loss,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
