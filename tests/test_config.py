"""Smoke tests for the YAML config loaders."""

from __future__ import annotations

import pytest

from minitron_ssm.utils import config as cfg


def test_load_base():
    base = cfg.load_base()
    assert base.parent_arch.layers == 52
    assert base.parent_arch.embedding == 4096
    assert base.parent_arch.mamba_heads == 128
    assert base.hardware.precision == "bf16"


def test_load_data():
    data = cfg.load_data()
    assert len(data.mixture) >= 2
    total = sum(m.weight for m in data.mixture)
    assert pytest.approx(total, rel=1e-6) == 1.0
    assert data.validation.num_tokens >= 10_000_000


def test_load_importance():
    imp = cfg.load_importance()
    assert imp.calibration.num_samples == 1024
    assert imp.mamba.score_source == "Wx"
    assert imp.mamba.preserve_group_structure is True


def test_load_search_space():
    space = cfg.load_search_space()
    assert space.max_candidates == 20
    assert len(space.anchors) >= 2
    assert 3072 in space.grids.embedding


def test_load_kd():
    kd = cfg.load_kd()
    assert 0.0 < kd.objective.alpha <= 1.0
    assert kd.budget.tokens_per_candidate == 15_000_000
    assert kd.budget.smoke_test_tokens == 2_000_000
    assert kd.optimizer.name == "adamw"


def test_load_eval():
    ev = cfg.load_eval()
    assert "hellaswag" in ev.lm_eval_harness.tasks
    assert ev.throughput.measure_iters > 0


def test_unknown_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("totally_unknown: 1\n")
    with pytest.raises(ValueError, match="Unknown keys"):
        cfg.load_base(bad)
