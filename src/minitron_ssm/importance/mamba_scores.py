"""Group-aware Mamba head + head-channel importance scoring.

Reference: plan sections 6.2 - 6.3.

Key constraints from the paper:

* Mamba heads must be ranked **within each SSM group**; cross-group
  permutation breaks the broadcast pattern.
* Mamba head channels share a **single ranking across all heads**;
  each channel index is preserved or pruned uniformly.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import torch

from ..models.inspect import MambaLayout


@dataclass
class MambaScores:
    """Per-layer Mamba importance scores."""

    # layer_idx -> tensor of shape (n_groups, heads_per_group)
    head_scores_per_group: Dict[int, Any]
    # layer_idx -> tensor of shape (head_channels,)
    channel_scores: Dict[int, Any]


def score_heads_and_channels(
    activations: Dict[str, Any],
    layout: MambaLayout,
    *,
    score_source: str = "Wx",
) -> MambaScores:
    """Compute group-aware head scores and shared channel scores.

    Parameters
    ----------
    activations:
        Output of :class:`ActivationCollector.aggregate`. Expected to
        contain Wx-input or Wx-output activations per Mamba layer
        (paper Table 2 picks ``Wx``; see plan section 6.2).
    layout:
        Per-layer Mamba metadata from :func:`describe_mamba_layout`.
    score_source:
        One of ``Wx``, ``Wz``, ``WO``. Default ``Wx`` follows the paper.

    """
    import torch

    source = score_source.lower()
    if source not in {"wx", "wz", "wo"}:
        raise ValueError(f"Unsupported score_source={score_source!r}; expected Wx/Wz/WO")

    def _pick_activation(layer_idx: int):
        match_tokens = [f".{layer_idx}.", f"layers.{layer_idx}", f"layer_{layer_idx}"]
        for key, val in activations.items():
            lkey = key.lower()
            if source in lkey and any(tok in lkey for tok in match_tokens):
                return val
        for key, val in activations.items():
            lkey = key.lower()
            if any(tok in lkey for tok in match_tokens):
                return val
        return None

    head_scores_per_group: Dict[int, Any] = {}
    channel_scores: Dict[int, Any] = {}

    for spec in layout.layers:
        vec = _pick_activation(spec.layer_idx)
        if vec is None:
            continue
        vec = torch.as_tensor(vec, dtype=torch.float32).flatten()
        d_inner = spec.n_heads * spec.head_channels
        if d_inner <= 0 or vec.numel() < d_inner:
            continue
        if spec.n_groups <= 0 or spec.n_heads % spec.n_groups != 0:
            continue

        x = vec[:d_inner].view(spec.n_heads, spec.head_channels)
        hpg = spec.n_heads // spec.n_groups
        # Head score: mean absolute activation over channels.
        head_scores = x.abs().mean(dim=-1).view(spec.n_groups, hpg)
        # Channel score: shared ranking across heads.
        chan_scores = x.abs().mean(dim=0)
        head_scores_per_group[spec.layer_idx] = head_scores
        channel_scores[spec.layer_idx] = chan_scores

    return MambaScores(
        head_scores_per_group=head_scores_per_group,
        channel_scores=channel_scores,
    )


def select_top_heads_per_group(
    scores: MambaScores, target_heads: int, n_groups: int
) -> Dict[int, List[int]]:
    """Pick the top ``target_heads / n_groups`` heads inside each group.

    Returns ``layer_idx -> list[head_index]`` where indices are *global*
    (within the layer, i.e. 0..n_heads-1) after concatenating the
    per-group selections in group order.

    """
    import torch

    if n_groups <= 0:
        raise ValueError("n_groups must be > 0")
    if target_heads <= 0 or target_heads % n_groups != 0:
        raise ValueError(
            f"target_heads must be positive and divisible by n_groups; got {target_heads}/{n_groups}"
        )
    keep_per_group = target_heads // n_groups
    out: Dict[int, List[int]] = {}
    for layer_idx, hs in scores.head_scores_per_group.items():
        h = torch.as_tensor(hs, dtype=torch.float32)
        if h.ndim != 2 or h.size(0) != n_groups:
            continue
        if keep_per_group > h.size(1):
            raise ValueError(
                f"Requested keep_per_group={keep_per_group} exceeds available heads per group={h.size(1)}"
            )
        keep: List[int] = []
        for g in range(n_groups):
            idx = torch.topk(h[g], k=keep_per_group, largest=True).indices.tolist()
            base = g * h.size(1)
            keep.extend(sorted(base + int(i) for i in idx))
        out[layer_idx] = keep
    return out


def select_top_channels(scores: MambaScores, target_channels: int) -> Dict[int, List[int]]:
    """Pick the top-``target_channels`` indices from the shared channel ranking.

    """
    import torch

    if target_channels <= 0:
        raise ValueError("target_channels must be > 0")
    out: Dict[int, List[int]] = {}
    for layer_idx, cs in scores.channel_scores.items():
        c = torch.as_tensor(cs, dtype=torch.float32).flatten()
        if target_channels > c.numel():
            raise ValueError(
                f"Requested target_channels={target_channels} exceeds available={c.numel()} for layer {layer_idx}"
            )
        idx = torch.topk(c, k=target_channels, largest=True).indices.tolist()
        out[layer_idx] = sorted(int(i) for i in idx)
    return out
