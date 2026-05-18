"""FFN width pruning.

Reference: plan section 7 ("FFN Pruning").

TODO(stage-3): real implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch.nn as nn


def prune_ffn_neurons(layer: "nn.Module", keep_neurons: List[int]) -> None:
    """Reduce FFN intermediate width by keeping ``keep_neurons`` indices.

    TODO(stage-3):
        * Slice ``gate_up_proj`` output rows. For SwiGLU the rows are
          stacked ``[gate; up]``; both halves use the same ``keep_neurons``.
        * Slice ``down_proj`` input columns by ``keep_neurons``.
        * Update ``layer.intermediate_size`` (or equivalent config field).
    """
    raise NotImplementedError("TODO(stage-3): FFN width pruning")
