"""Stage 8: short KD on top-3 candidates (~200M tokens each).

Reference: plan section 9.2.
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

log = get_logger("kd-train")


def _train_one(base, data, kd, teacher, tokenizer, rec) -> dict:
    """Run KD on a single candidate and return its result record."""
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

    stream = mixed_stream(data, seed=base.seed)
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
        f"{cand_cfg.get('id', rec['candidate_id'])}-kd",
    )
    log.info("KD complete for %s", cand_cfg.get("id", rec["candidate_id"]))
    return {
        "candidate_id": cand_cfg.get("id", rec["candidate_id"]),
        "checkpoint": str(save_path),
        "tokens_seen": state.tokens_seen,
        "steps": state.step,
        "last_loss": state.last_loss,
    }


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _merge_partials(eval_dir: Path) -> Path:
    """Merge per-candidate result files into the aggregate ``08_kd_results.json``.

    Per-candidate files (``08_kd_results.cand<NN>.json``) are written by each
    one-candidate-per-job array task; this orders them by candidate index so
    ``08_kd_results.json[0]`` stays the top-ranked candidate (matching the order
    in ``06_top3.json``) for stage 09.
    """
    parts: list[tuple[int, dict]] = []
    for f in sorted(eval_dir.glob("08_kd_results.cand*.json")):
        m = re.search(r"cand(\d+)\.json$", f.name)
        if not m:
            continue
        parts.append((int(m.group(1)), json.loads(f.read_text(encoding="utf-8"))))
    parts.sort(key=lambda x: x[0])
    out_path = eval_dir / "08_kd_results.json"
    _atomic_write_json(out_path, [rec for _, rec in parts])
    return out_path


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument(
        "--candidate-index",
        type=int,
        default=None,
        help=(
            "Train only the Nth top candidate (0-based) and write a per-candidate "
            "result file, for one-candidate-per-job Slurm arrays. Omit to train all "
            "top candidates sequentially in a single process."
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
    top_path = eval_dir / "06_top3.json"
    if not top_path.exists():
        raise FileNotFoundError(f"missing {top_path}; run scripts/06_select_top3.py first")
    top = json.loads(top_path.read_text(encoding="utf-8"))[: int(kd.budget.top_candidates)]
    if not top:
        raise ValueError("06_top3.json is empty")

    teacher, tokenizer = load_parent(base, eval_mode=True)

    if args.candidate_index is not None:
        idx = args.candidate_index
        if not 0 <= idx < len(top):
            raise IndexError(
                f"--candidate-index {idx} out of range for {len(top)} top candidates"
            )
        rec = top[idx]
        log.info("training candidate index %d (%s)", idx, rec["candidate_id"])
        result = _train_one(base, data, kd, teacher, tokenizer, rec)
        partial_path = eval_dir / f"08_kd_results.cand{idx:02d}.json"
        _atomic_write_json(partial_path, result)
        log.info("wrote %s", partial_path)
        merged = _merge_partials(eval_dir)
        log.info("merged per-candidate results into %s", merged)
        return 0

    output = [_train_one(base, data, kd, teacher, tokenizer, rec) for rec in top]
    out_path = eval_dir / "08_kd_results.json"
    _atomic_write_json(out_path, output)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
