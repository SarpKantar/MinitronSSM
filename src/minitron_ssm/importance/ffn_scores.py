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
    # Hold multiple candidates per layer, then pick the best signal.
    by_layer: Dict[int, list[tuple[int, Any]]] = {}

    def _priority(key: str) -> int:
        k = key.lower()
        # Best source: FFN expansion activations.
        if any(tok in k for tok in ("gate_up_proj", "up_proj", "fc1", "gate_proj")):
            return 3
        # Accept generic FFN/MLP internals.
        if any(tok in k for tok in ("ffn", "mlp", "intermediate")):
            return 2
        # Down projection usually has hidden-size output; lowest priority.
        if any(tok in k for tok in ("down_proj", "fc2", "proj_down")):
            return 1
        return 0

    def _maybe_fold_swiglu(vec: Any, key: str) -> Any:
        k = key.lower()
        n = int(vec.numel())
        if n <= 1 or n % 2 != 0:
            return vec
        # Only fold when the module is a single *combined* gate+up projection.
        # Separate gate_proj or up_proj each already have the correct ffn width;
        # folding them would wrongly halve their size.
        if "gate_up_proj" in k or "gate_up" in k:
            half = n // 2
            return 0.5 * (vec[:half] + vec[half:])
        return vec

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
        t = _maybe_fold_swiglu(t.flatten(), key)
        by_layer.setdefault(layer_idx, []).append((_priority(key), t))

    for layer_idx, candidates in by_layer.items():
        # Pick highest priority first; then the widest vector.
        best_prio = max(p for p, _ in candidates)
        same_prio = [v for p, v in candidates if p == best_prio]
        best = max(same_prio, key=lambda x: int(x.numel()))
        out[layer_idx] = best

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
            # Some layers (e.g. attention-only blocks) may not expose FFN
            # vectors with target width; skip them rather than failing all pruning.
            continue
        idx = torch.topk(t, k=target_ffn, largest=True).indices.tolist()
        out[layer_idx] = sorted(int(i) for i in idx)
    if not out:
        raise ValueError(
            f"No FFN layers had at least target_ffn={target_ffn} neurons in scores"
        )
    return out
