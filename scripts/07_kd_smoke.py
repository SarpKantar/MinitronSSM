"""Stage 7: short KD smoke test (~5-20M tokens) before committing to full runs.

Reference: plan section 9.2.
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import base_parser, print_resolved

from minitron_ssm.data.mixture import mixed_stream
from minitron_ssm.data.tokenize import packed_token_batches
from minitron_ssm.kd.trainer import KDTrainer
from minitron_ssm.models.load import (
    build_pruned_model,
    disable_fast_mamba_kernels,
    load_parent,
)
from minitron_ssm.utils.checkpoint import load_candidate, save_candidate
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

    top_path = Path(base.paths.eval_dir) / "06_top3.json"
    if not top_path.exists():
        raise FileNotFoundError(f"missing {top_path}; run scripts/06_select_top3.py first")
    top = json.loads(top_path.read_text(encoding="utf-8"))
    if not top:
        raise ValueError("06_top3.json is empty")
    best = top[0]

    teacher, tokenizer = load_parent(base, eval_mode=True)
    state_dict, cand_cfg = load_candidate(Path(best["checkpoint"]))
    student = build_pruned_model(teacher, cand_cfg, eval_mode=False)
    student.load_state_dict(state_dict, strict=True)
    patched = disable_fast_mamba_kernels(student)
    if patched:
        log.info("disabled fused Mamba kernels for %d module(s) in KD student", patched)
    device = next(teacher.parameters()).device
    student = student.to(device)
    student.train()

    stream = mixed_stream(data, seed=base.seed)
    train_loader = packed_token_batches(
        stream,
        tokenizer=tokenizer,
        seq_len=kd.training.seq_len,
        batch_size=max(1, int(kd.training.micro_batch_size)),
    )
    trainer = KDTrainer(student, teacher, kd, train_loader, val_loader=None)
    state = trainer.train(target_tokens=int(kd.budget.smoke_test_tokens))

    out_dir = Path(base.paths.checkpoints_dir)
    save_path = save_candidate(
        student.state_dict(),
        cand_cfg,
        out_dir,
        f"{cand_cfg.get('id', best['candidate_id'])}-kd-smoke",
    )
    summary_path = Path(base.paths.eval_dir) / "07_kd_smoke.json"
    payload = {
        "candidate_id": cand_cfg.get("id", best["candidate_id"]),
        "checkpoint": str(save_path),
        "tokens_seen": state.tokens_seen,
        "steps": state.step,
        "last_loss": state.last_loss,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
