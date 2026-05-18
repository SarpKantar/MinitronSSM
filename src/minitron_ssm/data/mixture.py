"""Weighted dataset mixture sampler.

Reference: plan section 4.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from .streaming import stream_dataset
from ..utils.config import DataConfig


def mixed_stream(data_cfg: DataConfig, seed: int = 0) -> Iterator[Any]:
    """Yield examples sampled from the configured mixture weights.
    """
    rng = np.random.default_rng(seed)
    entries = list(data_cfg.mixture)
    if not entries:
        raise ValueError("data_cfg.mixture is empty")

    weights = np.asarray([max(0.0, e.weight) for e in entries], dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("all mixture weights are zero")
    weights = weights / weights.sum()

    streams = [
        stream_dataset(e.name, e.hf_path, e.hf_subset, split="train")
        for e in entries
    ]
    while True:
        idx = int(rng.choice(len(entries), p=weights))
        try:
            yield next(streams[idx])
        except StopIteration:
            # Re-open exhausted stream lazily to keep an infinite sampler.
            e = entries[idx]
            streams[idx] = stream_dataset(e.name, e.hf_path, e.hf_subset, split="train")
            yield next(streams[idx])
