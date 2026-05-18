"""FFN neuron importance scoring.

Reference: plan section 6.4.

"""

from __future__ import annotations

import re
from typing import Any, Dict


def score_ffn_neurons(activations: Dict[str, Any]) -> Dict[int, Any]:
    """Return ``layer_idx -> tensor of shape (ffn,)``.
    """
    import torch

    out: Dict[int, Any] = {}
    patt = re.compile(r"(?:layers|layer)[._](\d+)")
    for key, val in activations.items():
        m = patt.search(key)
        if not m:
            continue
        layer_idx = int(m.group(1))
        t = torch.as_tensor(val, dtype=torch.float32)
        if t.ndim > 1:
            t = t.abs().mean(dim=tuple(range(t.ndim - 1)))
        else:
            t = t.abs()
        out[layer_idx] = t.flatten()
    return out


def select_top_neurons(
    scores: Dict[int, Any], target_ffn: int
) -> Dict[int, "list[int]"]:
    """Pick the top-``target_ffn`` neuron indices per FFN layer.
    """
    import torch

    if target_ffn <= 0:
        raise ValueError("target_ffn must be > 0")
    out: Dict[int, list[int]] = {}
    for layer_idx, s in scores.items():
        t = torch.as_tensor(s, dtype=torch.float32).flatten()
        if target_ffn > t.numel():
            raise ValueError(
                f"target_ffn={target_ffn} exceeds available={t.numel()} in layer {layer_idx}"
            )
        idx = torch.topk(t, k=target_ffn, largest=True).indices.tolist()
        out[layer_idx] = sorted(int(i) for i in idx)
    return out
