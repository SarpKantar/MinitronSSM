"""Online KD trainer for the short KD recovery stage.

Reference: plan section 9.3.

TODO(stage-7/8): real implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

from .losses import kd_blended_loss
from ..utils.logging import get_logger
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
        import torch

        self.student = student
        self.teacher = teacher
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.state = KDState(step=0, tokens_seen=0, last_loss=float("nan"))
        self.log = get_logger("kd-trainer")
        self.grad_accumulation = self._resolve_grad_accumulation(cfg)

        if cfg.training.activation_checkpointing and hasattr(
            self.student, "gradient_checkpointing_enable"
        ):
            self.student.gradient_checkpointing_enable()

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(cfg.optimizer.lr),
            weight_decay=float(cfg.optimizer.weight_decay),
            betas=tuple(cfg.optimizer.betas),
        )
        # Scheduler is built lazily in ``train`` once the total number of
        # optimizer steps is known (it depends on the token budget).
        self.scheduler = None

    def _resolve_grad_accumulation(self, cfg: KDConfig) -> int:
        ga = cfg.training.grad_accumulation
        if isinstance(ga, int):
            return max(1, ga)
        if isinstance(ga, str) and ga.lower() == "auto":
            gbs = max(1, int(cfg.training.global_batch_size))
            mbs = max(1, int(cfg.training.micro_batch_size))
            return max(1, (gbs + mbs - 1) // mbs)
        raise ValueError(f"Unsupported grad_accumulation value: {ga!r}")

    def _build_scheduler(self, total_steps: int):
        """Warmup-then-cosine schedule that decays over the *whole* run.

        ``total_steps`` is the total number of optimizer updates (not
        micro-steps). The previous implementation tied the cosine decay
        horizon to ``warmup_steps`` and therefore drove the LR to its floor
        within ~``warmup_steps`` updates regardless of how long training
        actually ran.
        """
        import math
        import torch

        warmup_steps = self.cfg.scheduler.warmup_steps or 0
        floor = float(self.cfg.scheduler.min_lr_ratio)
        decay_steps = max(1, total_steps - warmup_steps)

        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / decay_steps
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + (1.0 - floor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_lambda)

    def train(self, target_tokens: int) -> KDState:
        """Train until ``target_tokens`` student tokens have been seen.

        """
        import math

        import torch

        if target_tokens <= 0:
            raise ValueError(f"target_tokens must be > 0, got {target_tokens}")

        # Total optimizer updates over the run, so the LR schedule spans the
        # whole budget instead of collapsing to the floor early.
        micro = max(1, int(self.cfg.training.micro_batch_size))
        seq_len = max(1, int(self.cfg.training.seq_len))
        tokens_per_update = max(1, self.grad_accumulation * micro * seq_len)
        total_opt_steps = max(1, math.ceil(target_tokens / tokens_per_update))
        self.scheduler = self._build_scheduler(total_opt_steps)
        self.log.info(
            "scheduler: %d optimizer steps (grad_accum=%d, tokens/update=%d, warmup=%d)",
            total_opt_steps,
            self.grad_accumulation,
            tokens_per_update,
            self.cfg.scheduler.warmup_steps or 0,
        )

        device = next(self.student.parameters()).device
        use_autocast = device.type == "cuda"
        self.teacher.eval()
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)

        while self.state.tokens_seen < target_tokens:
            batch = next(self.train_loader)
            if not isinstance(batch, dict):
                raise ValueError("train_loader must yield dict batches with input_ids/labels")
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", batch["input_ids"]).to(device)

            with torch.no_grad():
                with (
                    torch.cuda.amp.autocast(dtype=torch.bfloat16)
                    if use_autocast
                    else torch.autocast(device_type=device.type, enabled=False)
                ):
                    teacher_logits = self.teacher(
                        input_ids=input_ids, use_cache=False
                    ).logits

            with (
                torch.cuda.amp.autocast(dtype=torch.bfloat16)
                if use_autocast
                else torch.autocast(device_type=device.type, enabled=False)
            ):
                student_logits = self.student(
                    input_ids=input_ids, use_cache=False
                ).logits
                loss = kd_blended_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    targets=labels,
                    temperature=self.cfg.objective.temperature,
                    alpha=self.cfg.objective.alpha,
                )

            (loss / self.grad_accumulation).backward()

            self.state.step += 1
            batch_tokens = int((labels != -100).sum().item())
            self.state.tokens_seen += batch_tokens
            self.state.last_loss = float(loss.item())

            if self.state.step % self.grad_accumulation == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.state.step % 10 == 0:
                self.log.info(
                    "step=%d tokens=%d loss=%.6f lr=%.3e",
                    self.state.step,
                    self.state.tokens_seen,
                    self.state.last_loss,
                    self.optimizer.param_groups[0]["lr"],
                )

        # flush trailing grads if we exited mid-accumulation
        if self.state.step % self.grad_accumulation != 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        return self.state
