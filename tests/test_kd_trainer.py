"""Smoke tests for KDTrainer with tiny toy models."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minitron_ssm.kd.trainer import KDTrainer
from minitron_ssm.utils.config import load_kd


class TinyLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, use_cache=False):  # noqa: ARG002
        x = self.embed(input_ids)
        logits = self.proj(x)
        return type("Out", (), {"logits": logits})


def _toy_loader(vocab_size: int, batch: int, seq: int):
    while True:
        tokens = torch.randint(0, vocab_size, (batch, seq), dtype=torch.long)
        yield {"input_ids": tokens, "labels": tokens.clone()}


def test_kd_trainer_runs_and_updates_state():
    kd = load_kd()
    kd.training.global_batch_size = 2
    kd.training.micro_batch_size = 1
    kd.training.grad_accumulation = 1
    kd.training.activation_checkpointing = False
    kd.scheduler.warmup_steps = 2

    vocab = 32
    teacher = TinyLM(vocab, 16)
    student = TinyLM(vocab, 16)
    loader = _toy_loader(vocab_size=vocab, batch=2, seq=8)

    trainer = KDTrainer(student=student, teacher=teacher, cfg=kd, train_loader=loader)
    state = trainer.train(target_tokens=64)
    assert state.tokens_seen >= 64
    assert state.step > 0
    assert torch.isfinite(torch.tensor(state.last_loss))
