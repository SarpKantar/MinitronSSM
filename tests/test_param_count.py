"""Tests for the analytical parameter counter."""

from __future__ import annotations

from minitron_ssm.search.param_count import (
    ArchSpec,
    LayerCounts,
    count_params,
    parent_arch_from_base,
)
from minitron_ssm.utils.config import load_base


def test_parent_count_in_band():
    """The analytical count for Nemotron-H 8B should be in [6B, 12B].

    The formula is approximate (we don't yet know the exact Mamba/MLP
    split from the loaded model). The test sets a loose band so the
    counter can be calibrated later without breaking CI.
    """
    parent = parent_arch_from_base(load_base())
    n = count_params(parent)
    assert 6e9 <= n <= 12e9, f"parent param count out of band: {n / 1e9:.2f}B"


def test_monotonic_in_embedding():
    parent = parent_arch_from_base(load_base())
    small = ArchSpec(
        layers=parent.layers,
        embedding=3072,
        ffn=parent.ffn,
        mamba_heads=parent.mamba_heads,
        mamba_head_channels=parent.mamba_head_channels,
        mamba_groups=parent.mamba_groups,
        vocab_size=parent.vocab_size,
        layer_counts=parent.layer_counts,
    )
    assert count_params(small) < count_params(parent)


def test_monotonic_in_ffn():
    parent = parent_arch_from_base(load_base())
    small = ArchSpec(
        layers=parent.layers,
        embedding=parent.embedding,
        ffn=10_752,
        mamba_heads=parent.mamba_heads,
        mamba_head_channels=parent.mamba_head_channels,
        mamba_groups=parent.mamba_groups,
        vocab_size=parent.vocab_size,
        layer_counts=parent.layer_counts,
    )
    assert count_params(small) < count_params(parent)


def test_monotonic_in_mamba_heads():
    parent = parent_arch_from_base(load_base())
    small = ArchSpec(
        layers=parent.layers,
        embedding=parent.embedding,
        ffn=parent.ffn,
        mamba_heads=96,
        mamba_head_channels=parent.mamba_head_channels,
        mamba_groups=parent.mamba_groups,
        vocab_size=parent.vocab_size,
        layer_counts=parent.layer_counts,
    )
    assert count_params(small) < count_params(parent)


def test_layer_counts_total_must_match():
    parent = parent_arch_from_base(load_base())
    bad = ArchSpec(
        layers=parent.layers,
        embedding=parent.embedding,
        ffn=parent.ffn,
        mamba_heads=parent.mamba_heads,
        mamba_head_channels=parent.mamba_head_channels,
        mamba_groups=parent.mamba_groups,
        vocab_size=parent.vocab_size,
        layer_counts=LayerCounts(attention=4, mamba=10, mlp=10),  # sum != 52
    )
    try:
        count_params(bad)
    except ValueError as e:
        assert "layer_counts.total" in str(e)
    else:
        raise AssertionError("expected ValueError for layer-count mismatch")
