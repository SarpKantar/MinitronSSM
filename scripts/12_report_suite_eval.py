"""Focused seven-task evaluation for the final report.

This reproduces the stable zero-shot protocol used by Stage 9. Candidate
checkpoints use the validated safe Mamba path; the official 4B reference keeps
its native inference kernels so its scores are comparable to the earlier
successful parent evaluation.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from _common import REPO_ROOT, base_parser, print_resolved

from minitron_ssm.eval.harness import run_harness
from minitron_ssm.models.load import (
    build_pruned_model,
    disable_fast_mamba_kernels,
    load_parent,
    load_reference_4b,
)
from minitron_ssm.utils.checkpoint import load_candidate
from minitron_ssm.utils.config import load_base, load_eval
from minitron_ssm.utils.logging import get_logger

log = get_logger("report-suite-eval")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "minitron_report_json"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dir / f"{path.name}.{os.getpid()}.json"
    staged.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    last_error: OSError | None = None
    try:
        for attempt in range(8):
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                shutil.copyfile(staged, tmp)
                os.replace(tmp, path)
                return
            except OSError as exc:
                last_error = exc
                tmp.unlink(missing_ok=True)
                time.sleep(attempt + 1)
    finally:
        staged.unlink(missing_ok=True)
    raise OSError(f"failed to publish JSON output: {path}") from last_error


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    compact.pop("samples", None)
    return compact


def _load_target(args: Any, base: Any) -> tuple[Any, Any, str, str]:
    import torch

    if args.target == "reference":
        model, tokenizer = load_reference_4b(base)
        return model.eval(), tokenizer, base.models.reference_4b, "native"

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for checkpoint targets")
    checkpoint = _resolve(args.checkpoint)
    parent, tokenizer = load_parent(base, eval_mode=True, template_on_cpu=True)
    state_dict, candidate_cfg = load_candidate(checkpoint)
    model = build_pruned_model(parent, candidate_cfg, eval_mode=True)
    model.load_state_dict(state_dict, strict=True)
    patched = disable_fast_mamba_kernels(model)
    log.info("disabled fused Mamba kernels for %d candidate module(s)", patched)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    del parent, state_dict
    return model, tokenizer, str(checkpoint), "safe_torch"


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument(
        "--target",
        choices=("checkpoint", "reference"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", default="1")
    args = parser.parse_args()

    base = load_base(args.base_config)
    eval_cfg = load_eval()
    output = _resolve(args.output)
    plan = {
        "target": args.target,
        "checkpoint": str(_resolve(args.checkpoint)) if args.checkpoint else None,
        "label": args.label,
        "output": str(output),
        "tasks": list(eval_cfg.lm_eval_harness.tasks),
        "num_fewshot": int(eval_cfg.lm_eval_harness.num_fewshot),
        "batch_size": args.batch_size,
    }
    if args.dry_run:
        print_resolved("base", base)
        print_resolved("eval", eval_cfg)
        print(json.dumps(plan, indent=2))
        return 0

    payload: dict[str, Any] = {
        "schema_version": 1,
        "model": args.label,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **plan,
    }
    _write_json_atomic(output, payload)

    try:
        model, tokenizer, checkpoint, kernel_mode = _load_target(args, base)
        payload["checkpoint"] = checkpoint
        payload["kernel_mode"] = kernel_mode
        _write_json_atomic(output, payload)
        harness = run_harness(
            model,
            eval_cfg,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
        )
        payload["harness"] = _compact(harness)
        payload["status"] = "completed"
        payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json_atomic(output, payload)
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json_atomic(output, payload)
        raise

    log.info("wrote completed report evaluation to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
