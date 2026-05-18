"""Depth (layer) pruning -- ablation only.

Reference: plan section 12 (Ablation A). The paper finds width-only
pruning to be much better at 50% compression; this module is here so
we can run the depth-vs-width ablation, not as a primary axis.

TODO(stage-3 / ablation): real implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch.nn as nn


def prune_layers(model: "nn.Module", keep_layer_indices: List[int]) -> None:
    """Drop layers not in ``keep_layer_indices``.

    TODO(stage-3 / ablation):
        * Preserve all attention layers (paper does not prune attention).
        * Remove low-importance Mamba and MLP blocks.
        * Re-index ``model.backbone.layers`` and update
          ``model.config.num_hidden_layers``.
    """
    raise NotImplementedError("TODO(stage-3 / ablation): depth pruning")
