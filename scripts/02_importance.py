"""Stage 2: collect activations and compute importance scores.

Reference: plan section 6.
"""

from __future__ import annotations

from pathlib import Path

from _common import base_parser, print_resolved

from minitron_ssm.data.mixture import mixed_stream
from minitron_ssm.data.tokenize import packed_token_batches
from minitron_ssm.importance.embed_scores import score_hidden_channels
from minitron_ssm.importance.ffn_scores import score_ffn_neurons
from minitron_ssm.importance.hooks import ActivationCollector
from minitron_ssm.importance.mamba_scores import score_heads_and_channels
from minitron_ssm.models.inspect import describe_mamba_layout
from minitron_ssm.models.load import load_parent
from minitron_ssm.utils.config import load_base, load_data, load_importance
from minitron_ssm.utils.logging import get_logger

log = get_logger("importance")


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for saved importance scores (defaults to configs/importance.yaml).",
    )
    args = p.parse_args()

    base = load_base(args.base_config)
    data = load_data()
    imp = load_importance()

    if args.dry_run:
        print_resolved("base", base)
        print_resolved("data", data)
        print_resolved("importance", imp)
        return 0

    import torch

    model, tokenizer = load_parent(base, eval_mode=True)
    layout = describe_mamba_layout(model)
    device = next(model.parameters()).device

    target_tokens = ("mamba", "ssm", "ffn", "mlp", "up_proj", "gate", "down_proj")
    targets = [
        name
        for name, _ in model.named_modules()
        if name and any(tok in name.lower() for tok in target_tokens)
    ]
    if not targets:
        raise ValueError("No activation hook targets detected on model")

    seq_len = imp.calibration.seq_len or base.seq_len.preferred
    stream = mixed_stream(data, seed=base.seed)
    loader = packed_token_batches(
        stream,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=imp.calibration.batch_size,
    )

    with ActivationCollector(model, targets=targets) as collector:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= imp.calibration.num_samples:
                    break
                input_ids = batch["input_ids"].to(device)
                _ = model(input_ids=input_ids, use_cache=False)
                if (i + 1) % 64 == 0:
                    log.info("processed %d / %d calibration batches", i + 1, imp.calibration.num_samples)

    activations = collector.aggregate()
    mamba_scores = score_heads_and_channels(
        activations,
        layout,
        score_source=imp.mamba.score_source,
    )
    ffn_scores = score_ffn_neurons(activations)
    embed_scores = score_hidden_channels(activations)

    out_path = args.output or Path(imp.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mamba_head_scores_per_group": mamba_scores.head_scores_per_group,
            "mamba_channel_scores": mamba_scores.channel_scores,
            "ffn_scores": ffn_scores,
            "embed_scores": embed_scores,
            "metadata": {
                "score_source": imp.mamba.score_source,
                "calibration_samples": imp.calibration.num_samples,
                "seq_len": seq_len,
                "targets": targets,
            },
        },
        out_path,
    )
    log.info("Wrote importance scores to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
