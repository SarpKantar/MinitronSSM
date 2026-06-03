"""Stage 5: zero-shot LM loss + throughput for every pruned candidate.

Reference: plan section 8.3.
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import base_parser, print_resolved

from minitron_ssm.data.streaming import stream_validation
from minitron_ssm.data.tokenize import packed_token_batches
from minitron_ssm.eval.lm_loss import measure_lm_loss
from minitron_ssm.eval.throughput import measure_throughput
from minitron_ssm.models.load import build_pruned_model, load_parent
from minitron_ssm.utils.checkpoint import load_candidate
from minitron_ssm.utils.config import load_base, load_data, load_eval
from minitron_ssm.utils.logging import get_logger

log = get_logger("eval-candidates")


def main() -> int:
    p = base_parser(__doc__ or "")
    args = p.parse_args()

    base = load_base(args.base_config)
    data = load_data()
    ev = load_eval()

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("eval", ev)
        return 0

    parent, tokenizer = load_parent(base, eval_mode=True)
    ckpt_root = Path(base.paths.checkpoints_dir)
    if not ckpt_root.exists():
        raise FileNotFoundError(f"missing checkpoints dir: {ckpt_root}")

    records = []
    seq_len = ev.lm_loss.seq_len or base.seq_len.preferred
    for cand_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
        try:
            state_dict, cand_cfg = load_candidate(cand_dir)
        except FileNotFoundError:
            continue
        model = build_pruned_model(parent, cand_cfg, eval_mode=True)
        model.load_state_dict(state_dict, strict=True)
        device = next(parent.parameters()).device
        model = model.to(device)

        val_stream = stream_validation(data)
        val_loader = packed_token_batches(
            val_stream,
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=1,
        )
        lm = measure_lm_loss(
            model,
            val_loader,
            seq_len=seq_len,
            num_tokens=ev.lm_loss.num_tokens,
        )
        tp = measure_throughput(
            model,
            tokenizer,
            input_lens=ev.throughput.input_lens,
            output_lens=ev.throughput.output_lens,
            batch_sizes=ev.throughput.batch_sizes,
            warmup_iters=ev.throughput.warmup_iters,
            measure_iters=ev.throughput.measure_iters,
        )
        best_tp = max((x.tokens_per_second for x in tp), default=0.0)
        records.append(
            {
                "candidate_id": cand_cfg.get("id", cand_dir.name),
                "checkpoint": str(cand_dir),
                "lm_loss": lm.loss,
                "perplexity": lm.perplexity,
                "num_tokens": lm.num_tokens,
                "tokens_per_second": best_tp,
                "throughput": [x.__dict__ for x in tp],
            }
        )
        log.info(
            "evaluated %s lm=%.5f tps=%.2f",
            cand_cfg.get("id", cand_dir.name),
            lm.loss,
            best_tp,
        )
        import torch
        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = Path(base.paths.eval_dir) / "05_candidates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
