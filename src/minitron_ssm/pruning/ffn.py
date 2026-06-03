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

    Best-effort implementation:
        * Finds gate/up and down linear projections by name heuristics.
        * Applies SwiGLU-style row slicing when gate_up has 2*ffn rows.
        * Updates common intermediate-size attributes when present.
    """
    import torch
    import torch.nn as nn

    if not keep_neurons:
        raise ValueError("keep_neurons must be non-empty")
    keep = sorted(set(int(i) for i in keep_neurons))

    gate_up_mod = None
    down_mod = None
    for name, mod in layer.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        lname = name.lower()
        if gate_up_mod is None and any(
            tok in lname for tok in ("gate_up_proj", "up_proj", "fc1")
        ):
            gate_up_mod = mod
        if down_mod is None and any(
            tok in lname for tok in ("down_proj", "fc2", "proj_down")
        ):
            down_mod = mod

    if gate_up_mod is None or down_mod is None:
        return

    old_ffn = int(getattr(down_mod, "in_features", gate_up_mod.out_features))
    if max(keep) >= old_ffn:
        raise ValueError(
            f"keep index {max(keep)} out of bounds for FFN width {old_ffn}"
        )

    with torch.no_grad():
        # gate_up: usually [2*ffn, hidden] for SwiGLU.
        gu_w = gate_up_mod.weight.data
        if gu_w.size(0) == 2 * old_ffn:
            idx = keep + [i + old_ffn for i in keep]
            gate_up_mod.weight = nn.Parameter(gu_w[idx, :].contiguous())
            if gate_up_mod.bias is not None:
                gate_up_mod.bias = nn.Parameter(gate_up_mod.bias.data[idx].contiguous())
        elif gu_w.size(0) == old_ffn:
            gate_up_mod.weight = nn.Parameter(gu_w[keep, :].contiguous())
            if gate_up_mod.bias is not None:
                gate_up_mod.bias = nn.Parameter(
                    gate_up_mod.bias.data[keep].contiguous()
                )

        # down: [hidden, ffn] so slice input columns.
        if down_mod.weight.data.size(1) == old_ffn:
            down_mod.weight = nn.Parameter(
                down_mod.weight.data[:, keep].contiguous()
            )

    # Keep metadata in sync where possible.
    new_ffn = len(keep)
    for attr in ("intermediate_size", "ffn_dim", "d_ff", "inner_dim"):
        if hasattr(layer, attr):
            setattr(layer, attr, new_ffn)
