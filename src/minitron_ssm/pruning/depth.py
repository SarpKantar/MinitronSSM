"""Depth (layer) pruning -- ablation only.

Reference: plan section 12 (Ablation A). The paper finds width-only
pruning to be much better at 50% compression; this module is here so
we can run the depth-vs-width ablation, not as a primary axis.

TODO(stage-3 / ablation): real implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ..models.inspect import _locate_layer_container

if TYPE_CHECKING:
    import torch.nn as nn


def prune_layers(model: "nn.Module", keep_layer_indices: List[int]) -> None:
    """Drop layers not in ``keep_layer_indices``.

    Best-effort ablation utility: keeps only the requested indices from
    the model's detected layer container.
    """
    import torch.nn as nn

    if not keep_layer_indices:
        raise ValueError("keep_layer_indices must be non-empty")

    path, layers = _locate_layer_container(model)
    keep = sorted(set(int(i) for i in keep_layer_indices if 0 <= int(i) < len(layers)))
    if not keep:
        raise ValueError("No valid layer indices remain after filtering")

    new_layers = [layers[i] for i in keep]
    cur = model
    parts = path.split(".")
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], nn.ModuleList(new_layers))

    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(new_layers)
