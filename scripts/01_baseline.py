"""Stage 1: load parent Nemotron-H-8B, measure LM loss + throughput.

Reference: plan section 5.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from _common import base_parser, print_resolved

from minitron_ssm.data.streaming import stream_validation
from minitron_ssm.data.tokenize import packed_token_batches
from minitron_ssm.eval.lm_loss import measure_lm_loss
from minitron_ssm.eval.throughput import measure_throughput
from minitron_ssm.models.load import load_parent
from minitron_ssm.utils.config import load_base, load_data, load_eval
from minitron_ssm.utils.logging import get_logger

log = get_logger("baseline")


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write baseline JSON (defaults to outputs/eval/01_baseline.json).",
    )
    p.add_argument(
        "--val-batch-size",
        type=int,
        default=1,
        help="Validation batch size for LM loss.",
    )
    args = p.parse_args()

    base = load_base(args.base_config)
    data = load_data()
    ev = load_eval()

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("data", data)
        print_resolved("eval", ev)
        return 0

    model, tokenizer = load_parent(base, eval_mode=True)

    seq_len = ev.lm_loss.seq_len or base.seq_len.preferred
    val_stream = stream_validation(data)
    val_loader = packed_token_batches(
        val_stream,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=args.val_batch_size,
    )

    log.info("Measuring LM validation loss...")
    lm = measure_lm_loss(
        model,
        val_loader,
        seq_len=seq_len,
        num_tokens=ev.lm_loss.num_tokens,
    )
    log.info("LM loss=%.6f, ppl=%.3f, tokens=%d", lm.loss, lm.perplexity, lm.num_tokens)

    log.info("Measuring throughput/latency...")
    tput = measure_throughput(
        model,
        tokenizer,
        input_lens=ev.throughput.input_lens,
        output_lens=ev.throughput.output_lens,
        batch_sizes=ev.throughput.batch_sizes,
        warmup_iters=ev.throughput.warmup_iters,
        measure_iters=ev.throughput.measure_iters,
    )

    payload = {
        "model": base.models.parent,
        "precision": base.hardware.precision,
        "seq_len": seq_len,
        "lm_loss": asdict(lm),
        "throughput": [asdict(x) for x in tput],
    }
    out_path = args.output or (Path(base.paths.eval_dir) / "01_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Wrote baseline results to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
