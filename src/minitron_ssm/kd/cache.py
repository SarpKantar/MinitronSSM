"""Top-k teacher-logit cache (KD memory fallback path).

Reference: plan section 9.3 Risk 2.

The cache is built by running the teacher once over the KD dataset
and storing, per token position, the top-k ``(values, indices)`` of
the teacher's logits. Training then becomes a single-model
(student-only) loop driven by ``kd_topk_kl_loss``.

TODO(stage-7/8): real implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import torch


@dataclass
class TopKEntry:
    values: "torch.Tensor"   # (seq_len, k), bf16
    indices: "torch.Tensor"  # (seq_len, k), long
    tokens: "torch.Tensor"   # (seq_len,), long


class TopKLogitCache:
    """Append-only on-disk cache of top-k teacher logits.

    TODO(stage-7/8):
        * Use a sharded binary format (e.g. one file per ~100k examples)
          so we don't OOM on a single mmap.
        * Store ``values`` in bf16 (cast on read), ``indices`` in int32.
        * Expose ``.iter_batches(batch_size, seq_len)`` for the student loop.
    """

    def __init__(self, path: Path, k: int):
        self.path = path
        self.k = k

    def build(self, teacher: Any, loader: Iterator[Any]) -> None:
        raise NotImplementedError("TODO(stage-7/8): build top-k cache from teacher")

    def iter_batches(self, batch_size: int, seq_len: int) -> Iterator[TopKEntry]:
        raise NotImplementedError("TODO(stage-7/8): stream cached top-k batches")
