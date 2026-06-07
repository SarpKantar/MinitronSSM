"""Measure final-checkpoint LM loss, throughput, latency, and peak memory."""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _common import REPO_ROOT, base_parser, print_resolved

from minitron_ssm.data.streaming import stream_validation
from minitron_ssm.data.tokenize import packed_token_batches
from minitron_ssm.eval.lm_loss import measure_lm_loss
from minitron_ssm.eval.throughput import measure_throughput
from minitron_ssm.models.load import build_pruned_model, load_parent
from minitron_ssm.utils.checkpoint import load_candidate
from minitron_ssm.utils.config import load_base, load_data, load_eval
from minitron_ssm.utils.logging import get_logger

log = get_logger("report-efficiency")


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


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--num-loss-tokens",
        type=int,
        default=1_000_000,
        help="Held-out tokens used for LM-loss measurement.",
    )
    args = parser.parse_args()

    base = load_base(args.base_config)
    data_cfg = load_data()
    eval_cfg = load_eval()
    checkpoint = _resolve(args.checkpoint)
    output = _resolve(args.output)
    plan = {
        "model": args.label,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "num_loss_tokens": args.num_loss_tokens,
        "seq_len": eval_cfg.lm_loss.seq_len or base.seq_len.preferred,
        "throughput": asdict(eval_cfg.throughput),
        "kernel_mode": "native",
    }
    if args.dry_run:
        print_resolved("base", base)
        print_resolved("eval", eval_cfg)
        print(json.dumps(plan, indent=2))
        return 0

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **plan,
    }
    _write_json_atomic(output, payload)

    try:
        import torch

        parent, tokenizer = load_parent(base, eval_mode=True, template_on_cpu=True)
        state_dict, candidate_cfg = load_candidate(checkpoint)
        model = build_pruned_model(parent, candidate_cfg, eval_mode=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        del parent, state_dict

        seq_len = int(plan["seq_len"])
        val_loader = packed_token_batches(
            stream_validation(data_cfg),
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=1,
        )
        log.info("measuring LM loss over %d tokens", args.num_loss_tokens)
        lm_result = measure_lm_loss(
            model,
            val_loader,
            seq_len=seq_len,
            num_tokens=args.num_loss_tokens,
        )
        payload["lm_loss"] = asdict(lm_result)
        _write_json_atomic(output, payload)

        log.info("measuring throughput, latency, and peak memory")
        throughput = measure_throughput(
            model,
            tokenizer,
            input_lens=eval_cfg.throughput.input_lens,
            output_lens=eval_cfg.throughput.output_lens,
            batch_sizes=eval_cfg.throughput.batch_sizes,
            warmup_iters=eval_cfg.throughput.warmup_iters,
            measure_iters=eval_cfg.throughput.measure_iters,
        )
        payload["throughput_results"] = [asdict(item) for item in throughput]
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

    log.info("wrote completed efficiency results to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
