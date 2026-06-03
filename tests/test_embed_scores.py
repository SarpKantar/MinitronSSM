"""Tests for embedding/hidden-channel score extraction."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minitron_ssm.importance.embed_scores import score_hidden_channels


def test_score_hidden_channels_prefers_hidden_dim_over_ffn_dim():
    activations = {
        "model.layers.0.mlp.up_proj": torch.randn(21504),
        "model.layers.0.mlp.gate_up_proj": torch.randn(2 * 21504),
        "model.layers.0.mlp.down_proj": torch.randn(4096),
        "model.layers.0.norm": torch.randn(4096),
    }
    scores = score_hidden_channels(activations)
    assert scores.numel() == 4096


def test_score_hidden_channels_fallback_still_returns_vector():
    activations = {
        "model.layers.0.weird_path_a": torch.randn(3072),
        "model.layers.1.weird_path_b": torch.randn(3072),
    }
    scores = score_hidden_channels(activations)
    assert scores.numel() == 3072
