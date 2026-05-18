"""Tests for the strict shape-check helper."""

from __future__ import annotations

import pytest

from minitron_ssm.utils.shape_check import (
    CandidateArch,
    ShapeMismatch,
    assert_mamba_group_divisible,
    assert_shapes,
    expected_shapes_for,
)


def _arch():
    return CandidateArch(
        layers=52,
        embedding=3072,
        ffn=12288,
        mamba_heads=112,
        mamba_head_channels=64,
        mamba_groups=8,
        vocab_size=131072,
    )


def test_assert_shapes_passes_on_correct_state():
    arch = _arch()
    assert_shapes(expected_shapes_for(arch), arch)


def test_assert_shapes_fails_on_drift():
    arch = _arch()
    shapes = dict(expected_shapes_for(arch))
    shapes["embedding.weight"] = (131072, 4096)  # parent dim, not pruned
    with pytest.raises(ShapeMismatch, match="embedding.weight"):
        assert_shapes(shapes, arch)


def test_partial_shape_map_is_ok():
    arch = _arch()
    full = expected_shapes_for(arch)
    partial = {"block.ffn.gate_up_proj.weight": full["block.ffn.gate_up_proj.weight"]}
    assert_shapes(partial, arch)


def test_group_divisibility_pass():
    arch = _arch()
    assert_mamba_group_divisible(arch)


def test_group_divisibility_fail():
    arch = CandidateArch(
        layers=52,
        embedding=3072,
        ffn=12288,
        mamba_heads=100,  # not divisible by 8
        mamba_head_channels=64,
        mamba_groups=8,
        vocab_size=131072,
    )
    with pytest.raises(ShapeMismatch, match="group"):
        assert_mamba_group_divisible(arch)
