"""LM validation loss measurement.

Reference: plan sections 5.2, 11.

TODO(stage-1 / stage-5 / stage-9): real implementation.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class LMLossResult:
    loss: float
    perplexity: float
    num_tokens: int


def measure_lm_loss(
    model: "nn.Module",
    val_loader: Iterator,
    *,
    seq_len: int,
    num_tokens: int,
) -> LMLossResult:
    """Average negative log-likelihood over up to ``num_tokens`` validation tokens.

    Loader items can be either tensors ``(B, T)`` or dicts containing
    ``input_ids`` and optional ``labels``.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    device = next(model.parameters()).device
    use_autocast = device.type == "cuda"

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                labels = batch.get("labels", input_ids)
            else:
                input_ids = batch
                labels = batch

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=torch.bfloat16)
                if use_autocast
                else nullcontext()
            )
            with autocast_ctx:
                out = model(input_ids=input_ids, labels=labels, use_cache=False)

            if getattr(out, "loss", None) is not None:
                token_mask = (labels != -100)
                n_tokens = int(token_mask.sum().item())
                if n_tokens <= 0:
                    continue
                total_nll += float(out.loss.item()) * n_tokens
                total_tokens += n_tokens
            else:
                logits = out.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                    ignore_index=-100,
                )
                n_tokens = int((shift_labels != -100).sum().item())
                total_nll += float(loss.item())
                total_tokens += n_tokens

            if total_tokens >= num_tokens:
                break

    if total_tokens == 0:
        raise ValueError("No tokens were processed in measure_lm_loss")

    avg_loss = total_nll / total_tokens
    ppl = math.exp(min(avg_loss, 50.0))
    return LMLossResult(loss=avg_loss, perplexity=ppl, num_tokens=total_tokens)
