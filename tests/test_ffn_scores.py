"""Tests for FFN importance scoring selection logic."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minitron_ssm.importance.ffn_scores import score_ffn_neurons, select_top_neurons


def test_score_ffn_prefers_up_proj_over_down_proj():
    acts = {
        "model.layers.1.mlp.down_proj": torch.randn(4096),
        "model.layers.1.mlp.up_proj": torch.randn(21504),
    }
    out = score_ffn_neurons(acts)
    assert 1 in out
    assert out[1].numel() == 21504


def test_score_ffn_folds_swiglu_gate_up_vector():
    acts = {
        "model.layers.2.mlp.gate_up_proj": torch.randn(2 * 12288),
    }
    out = score_ffn_neurons(acts)
    assert out[2].numel() == 12288


def test_select_top_neurons_skips_layers_with_small_vectors():
    scores = {
        0: torch.randn(4096),
        1: torch.randn(12288),
    }
    keep = select_top_neurons(scores, target_ffn=10752)
    assert 0 not in keep
    assert 1 in keep and len(keep[1]) == 10752
