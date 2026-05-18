"""Knowledge-distillation loss functions.

Implements the objectives described in plan section 9.1:

* Temperature-scaled forward KL between student and teacher logits.
* A top-k variant that only retains the teacher's top-k tokens per
  position (memory-efficient; matches the paper's later SFT-KD stage
  which uses top-k teacher logits, plan section 9.1).
* A blended ``alpha * KD + (1 - alpha) * CE`` loss.

The implementation imports ``torch`` lazily inside each function so
this module can be imported by tests on a CPU-only dev box (torch is
required to *call* these functions, but not to import them).
"""

from __future__ import annotations

from typing import Optional


def kd_kl_loss(
    student_logits,
    teacher_logits,
    temperature: float = 1.0,
    reduction: str = "batchmean",
):
    """Forward KL with temperature scaling.

    ``loss = KL(softmax(teacher/T) || softmax(student/T)) * T**2``

    The ``T**2`` factor preserves gradient magnitudes (Hinton et al.).

    Both ``student_logits`` and ``teacher_logits`` must have shape
    ``(..., vocab)`` and identical leading dimensions.
    """
    import torch
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_logits.shape)} vs "
            f"teacher {tuple(teacher_logits.shape)}"
        )

    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)

    flat_s = s.reshape(-1, s.size(-1))
    flat_t = t.reshape(-1, t.size(-1))
    loss = F.kl_div(flat_s, flat_t, reduction=reduction)
    return loss * (temperature ** 2)


def kd_topk_kl_loss(
    student_logits,
    teacher_topk_values,
    teacher_topk_indices,
    temperature: float = 1.0,
    reduction: str = "batchmean",
):
    """Top-k forward KL between student logits and *truncated* teacher logits.

    Renormalises the teacher distribution over its top-k tokens and
    matches the student's softmax over the *same* top-k indices.
    This is a tight approximation of the full-vocab KL when ``k``
    captures most of the teacher's probability mass.

    Shapes:

    * ``student_logits``: ``(..., vocab)``
    * ``teacher_topk_values``: ``(..., k)`` -- raw logits (not softmax)
    * ``teacher_topk_indices``: ``(..., k)`` -- long
    """
    import torch
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if teacher_topk_values.shape != teacher_topk_indices.shape:
        raise ValueError("teacher topk values/indices shape mismatch")
    if teacher_topk_values.shape[:-1] != student_logits.shape[:-1]:
        raise ValueError("leading dims of teacher topk must match student")

    s_log = F.log_softmax(student_logits / temperature, dim=-1)
    s_log_topk = s_log.gather(-1, teacher_topk_indices)
    t = F.softmax(teacher_topk_values / temperature, dim=-1)

    flat_s = s_log_topk.reshape(-1, s_log_topk.size(-1))
    flat_t = t.reshape(-1, t.size(-1))
    loss = F.kl_div(flat_s, flat_t, reduction=reduction)
    return loss * (temperature ** 2)


def kd_blended_loss(
    student_logits,
    teacher_logits,
    targets,
    *,
    temperature: float = 1.0,
    alpha: float = 0.9,
    teacher_topk_values=None,
    teacher_topk_indices=None,
    ignore_index: int = -100,
):
    """``alpha * KD + (1 - alpha) * CE`` blended objective.

    If ``teacher_topk_values`` / ``teacher_topk_indices`` are provided
    they take precedence over ``teacher_logits`` and the top-k KL is
    used instead of the full-vocab KL.

    ``targets`` are the true next-token ids; positions equal to
    ``ignore_index`` are masked out of the CE term.
    """
    import torch
    import torch.nn.functional as F

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if teacher_topk_values is not None and teacher_topk_indices is not None:
        kd = kd_topk_kl_loss(
            student_logits,
            teacher_topk_values,
            teacher_topk_indices,
            temperature=temperature,
        )
    else:
        if teacher_logits is None:
            raise ValueError("either teacher_logits or teacher_topk_* must be provided")
        kd = kd_kl_loss(student_logits, teacher_logits, temperature=temperature)

    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
    return alpha * kd + (1.0 - alpha) * ce
