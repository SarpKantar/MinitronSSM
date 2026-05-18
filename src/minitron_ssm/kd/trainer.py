"""Online KD trainer for the short KD recovery stage.

Reference: plan section 9.3.

TODO(stage-7/8): real implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

from ..utils.config import KDConfig

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


@dataclass
class KDState:
    step: int
    tokens_seen: int
    last_loss: float


class KDTrainer:
    """Single-machine online KD trainer.

    Each step runs:
        teacher.eval()        -- no grad
        student.train()
        loss = blended_kd_loss(student(batch), teacher(batch), batch.targets)
        loss.backward()
        opt.step()

    TODO(stage-7/8):
        * Wire up AdamW + cosine schedule from ``cfg.optimizer`` / ``cfg.scheduler``.
        * Enable activation checkpointing on the student if
          ``cfg.training.activation_checkpointing``.
        * Derive ``grad_accumulation`` if set to ``"auto"``.
        * Use ``torch.cuda.amp.autocast(dtype=torch.bfloat16)``.
        * Stop after ``tokens_per_candidate`` tokens.
        * Periodically log validation LM loss.
    """

    def __init__(
        self,
        student: "nn.Module",
        teacher: "nn.Module",
        cfg: KDConfig,
        train_loader: Iterator[Any],
        val_loader: Optional[Iterator[Any]] = None,
    ):
        self.student = student
        self.teacher = teacher
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.state = KDState(step=0, tokens_seen=0, last_loss=float("nan"))

    def train(self, target_tokens: int) -> KDState:
        """Train until ``target_tokens`` student tokens have been seen.

        TODO(stage-7/8): main training loop.
        """
        raise NotImplementedError("TODO(stage-7/8): KD training loop")
