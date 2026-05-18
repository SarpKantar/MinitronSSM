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

    TODO(stage-3):
        * Slice the token embedding output dim.
        * Slice every Linear projection's input columns when its input
          is the hidden state, and output rows when its output is the
          hidden state.
        * Slice every RMSNorm / LayerNorm parameter.
        * Slice the LM-head input columns.
        * Update ``model.config.hidden_size``.
        * Run shape checks after each layer.
    """
    raise NotImplementedError("TODO(stage-3): global hidden-dim pruning")
