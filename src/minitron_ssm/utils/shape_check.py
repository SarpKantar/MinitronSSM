"""Strict shape assertions for pruned models.

Used as a safety net after every pruning step (Mamba heads, head
channels, FFN, embedding). The idea is to catch shape drift before
attempting a forward pass on the GPU, where errors are slow and
expensive to diagnose.

The implementation is intentionally framework-agnostic: it accepts a
plain ``dict[str, tuple[int, ...]]`` ("shape map") plus a
``CandidateArch`` dataclass, so it can be unit-tested on CPU without
any torch import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

ShapeMap = Mapping[str, Tuple[int, ...]]


@dataclass(frozen=True)
class CandidateArch:
    """Architecture-level dimensions for a single candidate.

    Mirrors the fields used by :func:`search.param_count.count_params`.
    """

    layers: int
    embedding: int
    ffn: int
    mamba_heads: int
    mamba_head_channels: int
    mamba_groups: int
    vocab_size: int


class ShapeMismatch(AssertionError):
    """Raised when a tensor in a (pruned) state dict has the wrong shape."""


def expected_shapes_for(arch: CandidateArch) -> Dict[str, Tuple[int, ...]]:
    """Return a minimal set of (tensor_name -> expected shape) entries.

    The names use logical placeholders rather than the real Nemotron-H
    parameter names; the real mapping is wired up in
    :mod:`minitron_ssm.pruning.apply` once we can introspect the model.

    Entries returned here cover the dimensions that pruning touches:

    * token embedding rows / hidden dim
    * FFN gate/up output rows and down-projection input columns
    * Mamba x/z input projections (heads x head_channels)
    * LM head input dim
    """
    h = arch.embedding
    d_mamba = arch.mamba_heads * arch.mamba_head_channels
    return {
        "embedding.weight": (arch.vocab_size, h),
        "lm_head.weight": (arch.vocab_size, h),
        "block.ffn.gate_up_proj.weight": (2 * arch.ffn, h),
        "block.ffn.down_proj.weight": (h, arch.ffn),
        "block.mamba.x_proj.weight": (d_mamba, h),
        "block.mamba.z_proj.weight": (d_mamba, h),
        "block.mamba.out_proj.weight": (h, d_mamba),
    }


def assert_shapes(shapes: ShapeMap, arch: CandidateArch) -> None:
    """Assert that *shapes* are consistent with *arch*.

    Raises :class:`ShapeMismatch` with a clear diff on first failure.
    Tensor names that are missing from *shapes* are skipped; only
    *mismatched* shapes are errors. This keeps the check usable with
    partial state dicts (e.g. just the pruning targets).
    """
    expected = expected_shapes_for(arch)
    errors: List[str] = []
    for name, want in expected.items():
        if name not in shapes:
            continue
        got = tuple(shapes[name])
        if got != want:
            errors.append(f"  {name}: expected {want}, got {got}")
    if errors:
        raise ShapeMismatch(
            "Shape mismatch vs. candidate arch:\n" + "\n".join(errors)
        )


def assert_mamba_group_divisible(arch: CandidateArch) -> None:
    """Group-aware constraint: heads must distribute evenly across groups.

    See plan section 6.3. Cross-group head permutation breaks the
    SSM broadcast pattern, so each group must keep the same number
    of heads.
    """
    if arch.mamba_heads % arch.mamba_groups != 0:
        raise ShapeMismatch(
            f"mamba_heads={arch.mamba_heads} not divisible by "
            f"mamba_groups={arch.mamba_groups}; group-aware pruning would "
            f"break the SSM broadcast pattern."
        )
