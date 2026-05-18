"""Mamba head + head-channel pruning.

Reference: plan section 7 ("Mamba Head Pruning",
"Mamba Head-Channel Pruning").

Affected tensors per Mamba layer (verify names with
:func:`models.inspect.describe_mamba_layout`):

* ``in_proj.weight`` rows for ``z``, ``x``, ``B``, ``C``, ``dt``
* ``conv1d.weight`` / ``conv1d.bias`` channels associated with x/B/C
* ``A_log``, ``D``, ``dt_bias`` (per-head)
* ``norm.weight`` (per-channel within d_inner)
* ``out_proj.weight`` input columns

TODO(stage-3): real implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch.nn as nn


def prune_mamba_heads(layer: "nn.Module", keep_idx_per_group: List[List[int]]) -> None:
    """Drop heads, keeping ``keep_idx_per_group[g]`` inside group ``g``.

    *In-place* mutation of the layer's parameter tensors. The number
    of kept heads must be the same across all groups (group-aware
    constraint, plan section 6.3).

    TODO(stage-3):
        * Build the flat head index list from per-group selections.
        * Slice ``in_proj`` rows for z, x, dt.
        * Slice B/C rows (per-group rather than per-head).
        * Slice ``A_log``, ``D``, ``dt_bias`` at the head dimension.
        * Slice ``conv1d`` channels for x.
        * Slice ``out_proj`` input columns at the head dimension.
        * Update ``layer.args`` / ``layer.config`` fields so later
          forwards see consistent dimensions.
    """
    raise NotImplementedError("TODO(stage-3): prune Mamba heads (group-aware)")


def prune_mamba_head_channels(layer: "nn.Module", keep_channels: List[int]) -> None:
    """Drop head channels uniformly across all heads in *layer*.

    TODO(stage-3):
        * Apply the same ``keep_channels`` index to every head when
          slicing z, x, ``conv1d`` x-channels, norm, and out_proj
          input columns.
        * Do NOT touch B/C/dt; head channels are independent of d_state.
    """
    raise NotImplementedError("TODO(stage-3): prune Mamba head-channels (shared)")
