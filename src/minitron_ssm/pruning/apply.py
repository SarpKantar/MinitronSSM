"""Orchestrator: turn a (parent, scores, candidate) triple into a pruned model.

Reference: plan sections 7-8.

TODO(stage-3): real implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..search.space import Candidate
from ..utils.shape_check import CandidateArch

if TYPE_CHECKING:
    import torch.nn as nn


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

    TODO(stage-3): wire up the four pruning passes plus shape check.
    Deep-copy the parent before mutating it, because we re-use the
    same parent across all candidates and KD as the teacher.
    """
    raise NotImplementedError("TODO(stage-3): orchestrate pruning for a candidate")


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
