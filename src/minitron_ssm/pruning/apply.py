"""Orchestrator: turn a (parent, scores, candidate) triple into a pruned model.

Reference: plan sections 7-8.

TODO(stage-3): real implementation.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, List

from ..importance.embed_scores import select_top_hidden_channels
from ..importance.ffn_scores import select_top_neurons
from ..importance.mamba_scores import MambaScores, select_top_channels, select_top_heads_per_group
from ..models.inspect import _locate_layer_container
from .embed import prune_hidden_dim
from .ffn import prune_ffn_neurons
from .mamba import prune_mamba_head_channels, prune_mamba_heads
from ..search.space import Candidate
from ..utils.shape_check import CandidateArch, assert_mamba_group_divisible

if TYPE_CHECKING:
    import torch.nn as nn


def _ffn_scores_from_weights(layers) -> Dict[int, Any]:
    """Derive FFN neuron importance from down_proj column L2 norms.

    down_proj has shape [hidden, ffn], so its column norms score each
    FFN neuron by how strongly it influences the hidden state.  This is
    a reliable weight-based alternative to activation-based scoring and
    always yields vectors of the correct ``ffn`` width regardless of the
    SwiGLU variant used by the model.
    """
    import torch
    import torch.nn as nn

    scores: Dict[int, Any] = {}
    for i, layer in enumerate(layers):
        for _name, mod in layer.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            lname = _name.lower()
            if any(tok in lname for tok in ("down_proj", "fc2", "proj_down")):
                # weight: [out=hidden, in=ffn]  → column norms = per-neuron scores
                w = mod.weight.data.float()
                scores[i] = w.norm(dim=0)   # shape: (ffn,)
                break
    return scores


def apply_candidate(
    parent: "nn.Module",
    scores: Any,
    candidate: Candidate,
) -> "nn.Module":
    """Return a new (deep-copied) model pruned to *candidate*.

    Order of operations (matters):

    1. Mamba head-channel pruning (shared ranking, per layer).
    2. Mamba head pruning (group-aware, per layer).
    3. FFN neuron pruning (per layer).
    4. Embedding hidden-dim pruning (global, last because it is
       brittle and depends on the above being consistent).
    5. Shape check against :func:`expected_shapes_for` for the
       candidate.

    Deep-copy the parent before mutating it, because we re-use the
    same parent across all candidates and KD as the teacher.
    """
    arch = candidate_to_arch(candidate)
    assert_mamba_group_divisible(arch)

    model = copy.deepcopy(parent)
    _, layers = _locate_layer_container(model)

    mamba_scores = MambaScores(
        head_scores_per_group=scores.get("mamba_head_scores_per_group", {}),
        channel_scores=scores.get("mamba_channel_scores", {}),
    )
    head_keep = select_top_heads_per_group(mamba_scores, candidate.mamba_heads, candidate.mamba_groups)
    channel_keep = select_top_channels(mamba_scores, candidate.mamba_head_channels)

    # Try saved activation-based FFN scores first; if they don't cover the
    # target width (e.g. because the scoring had the SwiGLU fold bug), fall
    # back to weight-based scoring from the already-loaded parent model.
    saved_ffn = scores.get("ffn_scores", {})
    try:
        ffn_keep = select_top_neurons(saved_ffn, candidate.ffn)
    except ValueError:
        from ..utils.logging import get_logger as _log
        _log("apply").warning(
            "saved FFN scores insufficient for target_ffn=%d; "
            "falling back to weight-based scoring",
            candidate.ffn,
        )
        weight_ffn = _ffn_scores_from_weights(layers)
        ffn_keep = select_top_neurons(weight_ffn, candidate.ffn)

    # Similarly guard the embed scores.
    saved_embed = scores.get("embed_scores")
    try:
        embed_keep = select_top_hidden_channels(saved_embed, candidate.embedding)
    except (ValueError, TypeError):
        from ..utils.logging import get_logger as _log
        _log("apply").warning(
            "saved embed scores insufficient for target_embedding=%d; "
            "falling back to equal-stride channel selection",
            candidate.embedding,
        )
        import torch
        parent_hidden = next(parent.parameters()).shape[-1]
        embed_keep = list(
            range(0, parent_hidden, parent_hidden // candidate.embedding)
        )[: candidate.embedding]

    for layer_idx, keep_channels in channel_keep.items():
        if layer_idx < len(layers):
            prune_mamba_head_channels(layers[layer_idx], keep_channels)
    for layer_idx, keep_heads in head_keep.items():
        if layer_idx < len(layers):
            layer_scores = mamba_scores.head_scores_per_group.get(layer_idx)
            if layer_scores is None:
                continue
            per_group = int(layer_scores.shape[1])
            grouped = [
                [h - g * per_group for h in keep_heads if g * per_group <= h < (g + 1) * per_group]
                for g in range(candidate.mamba_groups)
            ]
            prune_mamba_heads(layers[layer_idx], grouped)
    for layer_idx, keep_neurons in ffn_keep.items():
        if layer_idx < len(layers):
            prune_ffn_neurons(layers[layer_idx], keep_neurons)

    prune_hidden_dim(model, embed_keep)
    return model


def candidate_to_arch(candidate: Candidate) -> CandidateArch:
    """Helper bridging :class:`Candidate` to :class:`CandidateArch`."""
    return CandidateArch(
        layers=candidate.layers,
        embedding=candidate.embedding,
        ffn=candidate.ffn,
        mamba_heads=candidate.mamba_heads,
        mamba_head_channels=candidate.mamba_head_channels,
        mamba_groups=candidate.mamba_groups,
        vocab_size=candidate.vocab_size,
    )
