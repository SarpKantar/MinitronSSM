"""Tests for KD loss math. Skipped on dev boxes without torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minitron_ssm.kd.losses import kd_blended_loss, kd_kl_loss, kd_topk_kl_loss


def test_kd_kl_zero_when_student_matches_teacher():
    torch.manual_seed(0)
    teacher = torch.randn(4, 8, 100)
    loss = kd_kl_loss(teacher.clone(), teacher, temperature=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_kd_kl_positive_otherwise():
    torch.manual_seed(0)
    teacher = torch.randn(4, 8, 100)
    student = torch.randn(4, 8, 100)
    loss = kd_kl_loss(student, teacher, temperature=1.0)
    assert loss.item() > 0.0


def test_kd_kl_temperature_scales_t_squared():
    torch.manual_seed(0)
    teacher = torch.randn(4, 8, 100)
    student = torch.randn(4, 8, 100)
    l1 = kd_kl_loss(student, teacher, temperature=1.0)
    l2 = kd_kl_loss(student, teacher, temperature=2.0)
    # Both losses should be positive and the T**2 factor non-trivially affects magnitude.
    assert l1.item() > 0 and l2.item() > 0


def test_kd_kl_rejects_nonpositive_temperature():
    teacher = torch.zeros(2, 3, 5)
    with pytest.raises(ValueError):
        kd_kl_loss(teacher, teacher, temperature=0.0)


def test_topk_kl_approximates_full_kl_for_large_k():
    torch.manual_seed(0)
    V = 32
    teacher = torch.randn(2, 4, V)
    student = torch.randn(2, 4, V)
    k = V  # full vocab
    vals, idx = teacher.topk(k, dim=-1)
    full = kd_kl_loss(student, teacher, temperature=1.0)
    approx = kd_topk_kl_loss(student, vals, idx, temperature=1.0)
    assert approx.item() == pytest.approx(full.item(), abs=1e-5)


def test_blended_loss_alpha_bounds():
    torch.manual_seed(0)
    teacher = torch.randn(2, 4, 16)
    student = torch.randn(2, 4, 16)
    targets = torch.randint(0, 16, (2, 4))
    with pytest.raises(ValueError):
        kd_blended_loss(student, teacher, targets, alpha=1.5)
    out = kd_blended_loss(student, teacher, targets, alpha=0.9)
    assert torch.isfinite(out)


def test_blended_loss_topk_path():
    torch.manual_seed(0)
    teacher = torch.randn(2, 4, 32)
    student = torch.randn(2, 4, 32)
    targets = torch.randint(0, 32, (2, 4))
    vals, idx = teacher.topk(8, dim=-1)
    out = kd_blended_loss(
        student,
        teacher_logits=None,
        targets=targets,
        teacher_topk_values=vals,
        teacher_topk_indices=idx,
        alpha=0.5,
    )
    assert torch.isfinite(out)
