"""Stage 8b: a SECOND round of KD that resumes from the stage-08 outputs.

This is the "extra training" variant used when spare compute remains. It is a
near-copy of ``scripts/08_kd_train.py`` with three differences:

  1. Starting weights come from ``outputs/eval/08_kd_results.json`` (the
     ``-kd`` checkpoints) instead of ``06_top3.json`` (the pre-KD pruned shells),
     so each candidate continues training rather than starting over.
  2. A different data seed (``base.seed + 1000``) so round 2 sees *fresh* tokens
     -> ~30M unique tokens across the two rounds rather than 15M seen twice.
  3. Outputs are written with a ``round2`` / ``-kd2`` suffix so the original
     stage-08 artifacts are left intact.

It deliberately does NOT modify the validated ``scripts/08_kd_train.py`` or the
token budget in ``configs/kd.yaml`` (reuses ``tokens_per_candidate`` = 15M,
which already fit a 24h slot in ~19h during round 1).

Reference: plan section 20 ("With 2 extra days: recommended re-runs").
"""

from __future__ import annotations

import json
import os
import re
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

log = get_logger("kd-train-r2")

ROUND2_SEED_OFFSET = 1000


def _train_one(base, data, kd, teacher, tokenizer, rec) -> dict:
    """Continue KD on a single candidate from its stage-08 ``-kd`` checkpoint."""
    state_dict, cand_cfg = load_candidate(Path(rec["checkpoint"]))
    student = build_pruned_model(teacher, cand_cfg, eval_mode=False)
    student.load_state_dict(state_dict, strict=True)
    patched = disable_fast_mamba_kernels(student)
    if patched:
        log.info(
            "disabled fused Mamba kernels for %d module(s) in KD student %s",
            patched,
            cand_cfg.get("id", rec["candidate_id"]),
        )
    device = next(teacher.parameters()).device
    student = student.to(device)
    student.train()

    # Fresh data ordering for round 2 so we distil on new tokens.
    stream = mixed_stream(data, seed=base.seed + ROUND2_SEED_OFFSET)
    train_loader = packed_token_batches(
        stream,
        tokenizer=tokenizer,
        seq_len=kd.training.seq_len,
        batch_size=max(1, int(kd.training.micro_batch_size)),
    )
    trainer = KDTrainer(student, teacher, kd, train_loader, val_loader=None)
    state = trainer.train(target_tokens=int(kd.budget.tokens_per_candidate))
    save_path = save_candidate(
        student.state_dict(),
        cand_cfg,
        Path(base.paths.checkpoints_dir),
        f"{cand_cfg.get('id', rec['candidate_id'])}-kd2",
    )
    log.info("round-2 KD complete for %s", cand_cfg.get("id", rec["candidate_id"]))
    return {
        "candidate_id": cand_cfg.get("id", rec["candidate_id"]),
        "checkpoint": str(save_path),
        "resumed_from": str(rec["checkpoint"]),
        "tokens_seen": state.tokens_seen,
        "steps": state.step,
        "last_loss": state.last_loss,
    }


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _merge_partials(eval_dir: Path) -> Path:
    """Merge per-candidate round-2 result files into ``08_kd_results_round2.json``."""
    parts: list[tuple[int, dict]] = []
    for f in sorted(eval_dir.glob("08_kd_results_round2.cand*.json")):
        m = re.search(r"cand(\d+)\.json$", f.name)
        if not m:
            continue
        parts.append((int(m.group(1)), json.loads(f.read_text(encoding="utf-8"))))
    parts.sort(key=lambda x: x[0])
    out_path = eval_dir / "08_kd_results_round2.json"
    _atomic_write_json(out_path, [rec for _, rec in parts])
    return out_path


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument(
        "--candidate-index",
        type=int,
        default=None,
        help=(
            "Continue training only the Nth candidate (0-based) from "
            "08_kd_results.json, for one-candidate-per-job Slurm arrays. Omit to "
            "process all candidates sequentially."
        ),
    )
    args = p.parse_args()

    base = load_base(args.base_config)
    data = load_data()
    kd = load_kd()

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("kd", kd)
        return 0

    eval_dir = Path(base.paths.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    src_path = eval_dir / "08_kd_results.json"
    if not src_path.exists():
        raise FileNotFoundError(
            f"missing {src_path}; run scripts/08_kd_train.py (round 1) first"
        )
    results = json.loads(src_path.read_text(encoding="utf-8"))
    if not results:
        raise ValueError("08_kd_results.json is empty")

    teacher, tokenizer = load_parent(base, eval_mode=True)

    if args.candidate_index is not None:
        idx = args.candidate_index
        if not 0 <= idx < len(results):
            raise IndexError(
                f"--candidate-index {idx} out of range for {len(results)} candidates"
            )
        rec = results[idx]
        log.info(
            "round-2 training candidate index %d (%s) from %s",
            idx,
            rec["candidate_id"],
            rec["checkpoint"],
        )
        result = _train_one(base, data, kd, teacher, tokenizer, rec)
        partial_path = eval_dir / f"08_kd_results_round2.cand{idx:02d}.json"
        _atomic_write_json(partial_path, result)
        log.info("wrote %s", partial_path)
        merged = _merge_partials(eval_dir)
        log.info("merged per-candidate round-2 results into %s", merged)
        return 0

    output = [_train_one(base, data, kd, teacher, tokenizer, rec) for rec in results]
    out_path = eval_dir / "08_kd_results_round2.json"
    _atomic_write_json(out_path, output)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
