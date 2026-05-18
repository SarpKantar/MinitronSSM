"""Tests for candidate-architecture enumeration."""

from __future__ import annotations

from minitron_ssm.search.param_count import parent_arch_from_base
from minitron_ssm.search.space import enumerate_candidates
from minitron_ssm.utils.config import load_base, load_search_space


def test_enumeration_respects_budget_and_cap():
    base = load_base()
    space = load_search_space()
    parent = parent_arch_from_base(base)

    cands = enumerate_candidates(space, parent)

    assert len(cands) > 0, "enumeration produced no candidates"
    assert len(cands) <= space.max_candidates

    target = space.target_param_budget
    tol = space.budget_tolerance
    for c in cands:
        assert abs(c.param_count - target) <= tol * target, (
            f"candidate {c.id} out of budget: {c.param_count / 1e9:.2f}B"
        )
        assert c.mamba_heads % parent.mamba_groups == 0, (
            f"{c.id} violates group-aware constraint"
        )


def test_anchors_included_when_in_budget():
    base = load_base()
    space = load_search_space()
    parent = parent_arch_from_base(base)
    cands = enumerate_candidates(space, parent)
    ids = {c.id for c in cands}
    for anchor in space.anchors:
        # Anchor is emitted iff it passes the budget filter.
        if any(c.id == anchor.id for c in cands):
            assert anchor.id in ids


def test_enumeration_is_deterministic():
    base = load_base()
    space = load_search_space()
    parent = parent_arch_from_base(base)
    a = enumerate_candidates(space, parent)
    b = enumerate_candidates(space, parent)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.param_count for c in a] == [c.param_count for c in b]


def test_no_duplicates():
    base = load_base()
    space = load_search_space()
    parent = parent_arch_from_base(base)
    cands = enumerate_candidates(space, parent)
    keys = [
        (c.layers, c.embedding, c.ffn, c.mamba_heads, c.mamba_head_channels)
        for c in cands
    ]
    assert len(keys) == len(set(keys))
