"""Stage 9b: final eval using an explicit KD-results file.

This is a non-invasive variant of ``scripts/09_final_eval.py`` for follow-up
runs such as round-2 KD. It leaves the validated stage-09 script untouched while
allowing the KD checkpoint source, output path, and KD selection policy to be
chosen from the CLI.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from _common import REPO_ROOT, base_parser, print_resolved

from minitron_ssm.eval.harness import run_harness
from minitron_ssm.models.load import (
    build_pruned_model,
    disable_fast_mamba_kernels,
    load_parent,
)
from minitron_ssm.utils.checkpoint import load_candidate
from minitron_ssm.utils.config import load_base, load_eval
from minitron_ssm.utils.logging import get_logger

log = get_logger("final-eval-results")


def _resolve_path(path: Path) -> Path:
    """Return an absolute path under the repo root when given a relative one."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _read_bytes_with_retries(path: Path, *, attempts: int = 8) -> bytes:
    """Read raw bytes with retries for transient NFS/Lustre read errors."""
    last_err: BaseException | None = None
    for attempt in range(attempts):
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"missing required JSON artifact: {path}") from None
        except OSError as e:
            last_err = e
            if attempt < attempts - 1:
                time.sleep(1.0 * (attempt + 1))
    raise OSError(f"failed to read bytes from {path}") from last_err


def _stage_to_local(path: Path) -> Path:
    """Copy a small artifact to node-local scratch before reading it."""
    stage_root = Path(os.environ.get("TMPDIR", "/tmp")) / "minitron_json_stage"
    stage_root.mkdir(parents=True, exist_ok=True)
    dest = stage_root / f"{path.name}.{os.getpid()}"
    shutil.copyfile(path, dest)
    return dest


def _read_json(path: Path) -> Any:
    """Read JSON with retries and a node-local staging fallback."""
    resolved = _resolve_path(path)
    try:
        payload = _read_bytes_with_retries(resolved)
    except OSError:
        staged = _stage_to_local(resolved)
        try:
            payload = _read_bytes_with_retries(staged, attempts=3)
        except OSError as staged_err:
            raise OSError(
                f"failed to read JSON artifact after retries: {resolved}"
            ) from staged_err
        finally:
            staged.unlink(missing_ok=True)
    return json.loads(payload.decode("utf-8"))


def _compact_harness_result(harness: dict[str, Any]) -> dict[str, Any]:
    """Drop per-example records that make lm-eval JSON outputs hundreds of MB."""
    compact = dict(harness)
    compact.pop("samples", None)
    return compact


def _write_json_atomic(path: Path, obj: Any, *, attempts: int = 8) -> None:
    """Serialize locally, then atomically publish with retries for Lustre/NFS."""
    path = _resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stage_root = Path(os.environ.get("TMPDIR", "/tmp")) / "minitron_json_stage"
    stage_root.mkdir(parents=True, exist_ok=True)
    staged = stage_root / f"{path.name}.{os.getpid()}.json"
    staged.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

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


def _read_json_optional(path: Path) -> list[Any]:
    """Best-effort read; returns [] if the file is absent."""
    resolved = _resolve_path(path)
    try:
        data = _read_json(resolved)
    except FileNotFoundError:
        return []
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {resolved}, got {type(data).__name__}")
    return data


def _read_mini_kd_record(path: Path) -> dict[str, Any]:
    """Read a stage-10 mini-KD summary (single JSON object with ``checkpoint``)."""
    resolved = _resolve_path(path)
    data = _read_json(resolved)
    if not isinstance(data, dict):
        raise ValueError(
            f"expected a JSON object in {resolved}, got {type(data).__name__}"
        )
    if not data.get("checkpoint"):
        raise ValueError(f"{resolved} is missing required field 'checkpoint'")
    return data


def _select_kd(records: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if not records:
        raise ValueError("KD results file is empty")
    if policy == "first":
        return records[0]
    if policy == "last_loss":
        return min(records, key=lambda r: float(r.get("last_loss", float("inf"))))
    raise ValueError(f"unknown KD selection policy: {policy}")


def _build_student(parent, checkpoint: Path):
    """Reconstruct a pruned/KD student from a candidate checkpoint."""
    import torch

    state_dict, cand_cfg = load_candidate(checkpoint)
    student = build_pruned_model(parent, cand_cfg, eval_mode=True)
    student.load_state_dict(state_dict, strict=True)
    patched = disable_fast_mamba_kernels(student)
    if patched:
        log.info("disabled fused Mamba kernels for %d module(s)", patched)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = next(parent.parameters()).device
    return student.to(device).eval()


def _candidate_id(entry: dict[str, Any]) -> str | None:
    return entry.get("candidate_id") or entry.get("id")


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument(
        "--kd-results",
        type=Path,
        default=None,
        help=(
            "KD result artifact to evaluate. Defaults to "
            "<eval_dir>/08_kd_results_round2.json."
        ),
    )
    p.add_argument(
        "--top-results",
        type=Path,
        default=None,
        help="Top-pruned artifact. Defaults to <eval_dir>/06_top3.json.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <eval_dir>/09_final_eval_round2.json.",
    )
    p.add_argument(
        "--kd-selection",
        choices=("last_loss", "first"),
        default="last_loss",
        help="How to choose the KD checkpoint from --kd-results.",
    )
    p.add_argument(
        "--kd-label",
        default="best_kd_round2",
        help="Model label to use for the selected KD checkpoint.",
    )
    p.add_argument(
        "--skip-parent",
        action="store_true",
        help="Do not re-evaluate the parent model.",
    )
    p.add_argument(
        "--skip-pruned",
        action="store_true",
        help="Do not re-evaluate the best pre-KD pruned checkpoint.",
    )
    p.add_argument(
        "--skip-kd",
        action="store_true",
        help="Do not evaluate the checkpoint from --kd-results.",
    )
    p.add_argument(
        "--mini-kd-json",
        type=Path,
        default=None,
        help=(
            "Stage-10 mini-final KD summary JSON (e.g. outputs/eval/10_mini_kd.json). "
            "Adds an extra harness entry for that checkpoint."
        ),
    )
    p.add_argument(
        "--mini-kd-label",
        default="mini_final_kd",
        help="Model label for the --mini-kd-json checkpoint.",
    )
    p.add_argument(
        "--include-optional",
        action="store_true",
        help="Also run GSM8K / HumanEval / MBPP if time allows.",
    )
    p.add_argument(
        "--harness-batch-size",
        default=None,
        help=(
            "lm-eval batch size. Use a fixed integer (e.g. 1) to avoid auto-detect "
            "OOM on large Nemotron parents; defaults to configs/eval.yaml."
        ),
    )
    args = p.parse_args()

    base = load_base(args.base_config)
    ev = load_eval()
    eval_dir = _resolve_path(Path(base.paths.eval_dir))
    top_path = _resolve_path(args.top_results or (eval_dir / "06_top3.json"))
    kd_path = _resolve_path(args.kd_results or (eval_dir / "08_kd_results_round2.json"))
    out_path = _resolve_path(args.output or (eval_dir / "09_final_eval_round2.json"))

    mini_kd_path = _resolve_path(args.mini_kd_json) if args.mini_kd_json else None
    selected_kd: dict[str, Any] | None = None
    mini_kd: dict[str, Any] | None = None

    if not args.skip_kd:
        kd_records = _read_json_optional(kd_path)
        if kd_records:
            selected_kd = _select_kd(kd_records, args.kd_selection)
    if mini_kd_path is not None:
        mini_kd = _read_mini_kd_record(mini_kd_path)

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("eval", ev)
        print(
            json.dumps(
                {
                    "top_results": str(top_path),
                    "kd_results": str(kd_path),
                    "output": str(out_path),
                    "kd_selection": args.kd_selection,
                    "selected_kd": selected_kd,
                    "mini_kd_json": str(mini_kd_path) if mini_kd_path else None,
                    "mini_kd": mini_kd,
                    "include_parent": not args.skip_parent,
                    "include_pruned": not args.skip_pruned,
                    "include_kd": not args.skip_kd,
                    "include_optional": args.include_optional,
                    "harness_batch_size": args.harness_batch_size
                    or ev.lm_eval_harness.batch_size,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    import torch

    eval_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_kd and selected_kd is None:
        kd_records = _read_json(kd_path)
        if not isinstance(kd_records, list) or not kd_records:
            raise ValueError(f"KD results file is empty: {kd_path}")
        selected_kd = _select_kd(kd_records, args.kd_selection)
    if mini_kd_path is not None and mini_kd is None:
        mini_kd = _read_mini_kd_record(mini_kd_path)

    if args.skip_kd and mini_kd is None:
        raise ValueError(
            "nothing to evaluate: enable --mini-kd-json or omit --skip-kd with "
            "a valid --kd-results file"
        )

    parent, tokenizer = load_parent(
        base,
        eval_mode=True,
        template_on_cpu=args.skip_parent,
    )
    if args.skip_parent:
        log.info("loaded parent on CPU as architecture template only")
    harness_batch_size = (
        args.harness_batch_size
        if args.harness_batch_size is not None
        else ev.lm_eval_harness.batch_size
    )

    plan: list[dict[str, Any]] = []
    if not args.skip_parent:
        patched = disable_fast_mamba_kernels(parent)
        if patched:
            log.info("disabled fused Mamba kernels on parent for %d module(s)", patched)
        plan.append(
            {
                "name": "parent",
                "model": parent,
                "checkpoint": base.models.parent,
                "source": "base_config",
            }
        )
    if not args.skip_pruned:
        top = _read_json(top_path)
        if not top:
            raise ValueError(f"{top_path} is empty")
        plan.append(
            {
                "name": "best_pruned",
                "checkpoint": top[0]["checkpoint"],
                "candidate_id": _candidate_id(top[0]),
                "source": str(top_path),
                "selection": "first",
            }
        )
    if not args.skip_kd:
        if selected_kd is None:
            raise ValueError(f"no KD record selected from {kd_path}")
        plan.append(
            {
                "name": args.kd_label,
                "checkpoint": selected_kd["checkpoint"],
                "candidate_id": _candidate_id(selected_kd),
                "source": str(kd_path),
                "selection": args.kd_selection,
                "last_loss": selected_kd.get("last_loss"),
                "tokens_seen": selected_kd.get("tokens_seen"),
                "resumed_from": selected_kd.get("resumed_from"),
            }
        )
    if mini_kd is not None:
        plan.append(
            {
                "name": args.mini_kd_label,
                "checkpoint": mini_kd["checkpoint"],
                "candidate_id": _candidate_id(mini_kd),
                "source": str(mini_kd_path),
                "stage": "10_mini_kd",
                "last_loss": mini_kd.get("last_loss"),
                "tokens_seen": mini_kd.get("tokens_seen"),
            }
        )

    if not plan:
        raise ValueError("evaluation plan is empty")

    results = []
    for entry in plan:
        name = entry["name"]
        student = entry.get("model")
        built_here = student is None
        if built_here:
            log.info("building %s from %s", name, entry["checkpoint"])
            student = _build_student(parent, Path(entry["checkpoint"]))
        harness = run_harness(
            student,
            ev,
            include_optional=args.include_optional,
            tokenizer=tokenizer,
            batch_size=harness_batch_size,
        )
        result = {k: v for k, v in entry.items() if k != "model"}
        result["harness"] = _compact_harness_result(harness)
        results.append(result)
        log.info("evaluated %s", name)
        if name == "parent":
            log.info("offloading parent to CPU to free GPU for student eval")
            parent = parent.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if built_here:
            del student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _write_json_atomic(out_path, results)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
