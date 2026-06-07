"""Stage 10b: second optional mini-final KD pass.

This follow-up resumes from ``outputs/eval/10_mini_kd.json`` and writes separate
``round2`` artifacts, so the validated stage-10 script and its outputs are left
untouched. It is only useful after ``scripts/10_optional_mini_kd.py`` has
finished successfully.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from _common import REPO_ROOT, base_parser, print_resolved

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

log = get_logger("mini-final-kd-r2")

ROUND2_SEED_OFFSET = 2000


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _read_json_with_retries(path: Path, *, attempts: int = 8):
    path = _resolve_path(path)
    last_err: BaseException | None = None
    for attempt in range(attempts):
        try:
            with open(path, "rb") as f:
                return json.loads(f.read().decode("utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"missing required JSON artifact: {path}") from None
        except (OSError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < attempts - 1:
                time.sleep(1.0 * (attempt + 1))
    raise OSError(f"failed to read JSON artifact after retries: {path}") from last_err


def _write_json_atomic(path: Path, obj, *, attempts: int = 8) -> None:
    path = _resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(os.environ.get("TMPDIR", "/tmp")) / "minitron_json_stage"
    stage_root.mkdir(parents=True, exist_ok=True)
    staged = stage_root / f"{path.name}.{os.getpid()}.json"
    staged.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    last_err: BaseException | None = None
    try:
        for attempt in range(attempts):
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                shutil.copyfile(staged, tmp)
                os.replace(tmp, path)
                return
            except OSError as e:
                last_err = e
                tmp.unlink(missing_ok=True)
                if attempt < attempts - 1:
                    time.sleep(1.0 * (attempt + 1))
    finally:
        staged.unlink(missing_ok=True)
    raise OSError(f"failed to publish JSON output after retries: {path}") from last_err


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument(
        "--target-tokens",
        type=int,
        default=10_000_000,
        help="Additional KD tokens to train after stage 10.",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Stage-10 JSON artifact. Defaults to <eval_dir>/10_mini_kd.json.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON artifact. Defaults to <eval_dir>/10_mini_kd_round2.json.",
    )
    args = p.parse_args()

    base = load_base(args.base_config)
    data = load_data()
    kd = load_kd()
    eval_dir = _resolve_path(Path(base.paths.eval_dir))
    source_path = _resolve_path(args.source or (eval_dir / "10_mini_kd.json"))
    out_path = _resolve_path(args.output or (eval_dir / "10_mini_kd_round2.json"))

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("kd", kd)
        log.info("would resume from %s", source_path)
        log.info("would train for %d additional tokens", args.target_tokens)
        log.info("would write %s", out_path)
        return 0

    prev = _read_json_with_retries(source_path)
    if not isinstance(prev, dict):
        raise ValueError(f"{source_path} must contain a JSON object")
    if not prev.get("checkpoint"):
        raise ValueError(f"{source_path} does not contain a checkpoint field")

    teacher, tokenizer = load_parent(base, eval_mode=True)
    state_dict, cand_cfg = load_candidate(_resolve_path(Path(prev["checkpoint"])))
    student = build_pruned_model(teacher, cand_cfg, eval_mode=False)
    student.load_state_dict(state_dict, strict=True)
    patched = disable_fast_mamba_kernels(student)
    if patched:
        log.info("disabled fused Mamba kernels for %d module(s)", patched)
    device = next(teacher.parameters()).device
    student = student.to(device)
    student.train()

    stream = mixed_stream(data, seed=base.seed + ROUND2_SEED_OFFSET)
    train_loader = packed_token_batches(
        stream,
        tokenizer=tokenizer,
        seq_len=kd.training.seq_len,
        batch_size=max(1, int(kd.training.micro_batch_size)),
    )
    trainer = KDTrainer(student, teacher, kd, train_loader, val_loader=None)
    state = trainer.train(target_tokens=int(args.target_tokens))

    candidate_id = cand_cfg.get("id", prev.get("candidate_id", "candidate"))
    save_path = save_candidate(
        student.state_dict(),
        cand_cfg,
        Path(base.paths.checkpoints_dir),
        f"{candidate_id}-mini-final2",
    )
    payload = {
        "candidate_id": candidate_id,
        "checkpoint": str(save_path),
        "resumed_from": str(prev["checkpoint"]),
        "tokens_seen": state.tokens_seen,
        "previous_tokens_seen": prev.get("tokens_seen"),
        "approx_total_mini_tokens": int(prev.get("tokens_seen", 0))
        + int(state.tokens_seen),
        "steps": state.step,
        "last_loss": state.last_loss,
    }
    _write_json_atomic(out_path, payload)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
