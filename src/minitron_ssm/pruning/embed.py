"""Embedding (hidden-dim) pruning.

This is the *dangerous* pruning path (plan section 7 "Embedding
Hidden-Dimension Pruning"). The hidden dimension appears in every
projection, every norm, the token embedding output, and the LM-head
input. A single bad slice produces silent shape drift that only
surfaces deep inside the forward pass.

Strategy:

1. Build one global keep-index from
   :func:`importance.embed_scores.select_top_hidden_channels`.
2. Walk *every* parameter that consumes or produces the hidden dim
   and slice it consistently.
3. Run :func:`utils.shape_check.assert_shapes` after every layer.

TODO(stage-3): real implementation. Consider falling back to NVIDIA
Model Optimizer if available (plan section 7 "Preferred Route").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch.nn as nn


def prune_hidden_dim(model: "nn.Module", keep_channels: List[int]) -> None:
    """Reduce hidden_size by keeping ``keep_channels`` channels globally.

    Best-effort implementation for common HF module types:
        * Embedding: slice output channels.
        * Linear: slice rows/cols where in/out features match hidden size.
        * LayerNorm: slice parameters and normalized_shape.
        * LM head is handled by the same Linear rule.
    """
    import torch
    import torch.nn as nn

    if not keep_channels:
        raise ValueError("keep_channels must be non-empty")
    keep = sorted(set(int(i) for i in keep_channels))

    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        for _n, p in model.named_parameters():
            if p.ndim == 2 and "embed" in _n.lower():
                hidden_size = int(p.shape[1])
                break
    if hidden_size is None:
        hidden_size = max(keep) + 1
    if max(keep) >= hidden_size:
        raise ValueError(
            f"keep index {max(keep)} out of bounds for hidden_size={hidden_size}"
        )

    with torch.no_grad():
        for _name, mod in model.named_modules():
            lname = _name.lower()

            if isinstance(mod, nn.Embedding):
                if mod.weight.data.ndim == 2 and mod.weight.data.size(1) == hidden_size:
                    mod.weight = nn.Parameter(mod.weight.data[:, keep].contiguous())
                    mod.embedding_dim = len(keep)
                continue

            if isinstance(mod, nn.Linear):
                # Attention head dimensions (num_heads * head_dim) are NOT part
                # of the hidden residual stream, even when they happen to equal
                # hidden_size (square q/o projections). The plan does not prune
                # attention, so only slice the hidden-facing side:
                #   q/k/v_proj -> hidden is the INPUT (columns)
                #   o_proj     -> hidden is the OUTPUT (rows)
                # (mamba ``out_proj`` is excluded from the attention rule because
                # its output genuinely is the hidden dim.)
                is_attn_in = any(t in lname for t in ("q_proj", "k_proj", "v_proj"))
                is_attn_out = "o_proj" in lname and "out_proj" not in lname
                w = mod.weight.data
                if w.size(1) == hidden_size and not is_attn_out:
                    mod.weight = nn.Parameter(w[:, keep].contiguous())
                    mod.in_features = len(keep)
                    w = mod.weight.data
                if w.size(0) == hidden_size and not is_attn_in:
                    mod.weight = nn.Parameter(w[keep, :].contiguous())
                    mod.out_features = len(keep)
                    if mod.bias is not None and mod.bias.data.size(0) == hidden_size:
                        mod.bias = nn.Parameter(mod.bias.data[keep].contiguous())
                continue

            if isinstance(mod, nn.LayerNorm):
                if mod.weight is not None and mod.weight.data.size(0) == hidden_size:
                    mod.weight = nn.Parameter(mod.weight.data[keep].contiguous())
                if mod.bias is not None and mod.bias.data.size(0) == hidden_size:
                    mod.bias = nn.Parameter(mod.bias.data[keep].contiguous())
                norm_shape = mod.normalized_shape
                if (
                    isinstance(norm_shape, tuple)
                    and len(norm_shape) == 1
                    and norm_shape[0] == hidden_size
                ):
                    mod.normalized_shape = (len(keep),)
                continue

            # Custom norms: Nemotron-H uses RMSNorm variants that are NOT
            # ``nn.LayerNorm`` instances, so the branch above never matches and
            # the per-layer ``norm.weight`` / final ``norm_f.weight`` would keep
            # the original hidden width. Prune any module that exposes a 1-D
            # ``weight`` whose length equals hidden_size. Mixer-internal norms
            # (e.g. the gated Mamba RMSNorm) operate on the inner dimension and
            # therefore do not match hidden_size, so they are left untouched.
            w = getattr(mod, "weight", None)
            if (
                isinstance(w, nn.Parameter)
                and w.data.ndim == 1
                and w.data.size(0) == hidden_size
            ):
                mod.weight = nn.Parameter(w.data[keep].contiguous())
                b = getattr(mod, "bias", None)
                if (
                    isinstance(b, nn.Parameter)
                    and b.data.ndim == 1
                    and b.data.size(0) == hidden_size
                ):
                    mod.bias = nn.Parameter(b.data[keep].contiguous())
                if hasattr(mod, "normalized_shape"):
                    mod.normalized_shape = (len(keep),)
                if hasattr(mod, "hidden_size"):
                    mod.hidden_size = len(keep)

    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        model.config.hidden_size = len(keep)
