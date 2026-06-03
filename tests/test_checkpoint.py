"""Tests for candidate checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from minitron_ssm.utils.checkpoint import load_candidate, save_candidate


def test_save_and_load_candidate_roundtrip(tmp_path: Path):
    state = {
        "layer.weight": torch.randn(4, 8),
        "layer.bias": torch.randn(4),
    }
    cfg = {"id": "cand-test", "embedding": 8}
    out = save_candidate(state, cfg, tmp_path, "cand-test")
    loaded_state, loaded_cfg = load_candidate(out)
    assert loaded_cfg["id"] == "cand-test"
    assert set(loaded_state.keys()) == set(state.keys())
    assert loaded_state["layer.weight"].shape == state["layer.weight"].shape
