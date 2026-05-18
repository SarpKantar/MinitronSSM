"""HF datasets streaming wrapper.

Reference: plan section 4.
"""

from __future__ import annotations

from typing import Any, Iterator

from ..utils.config import DataConfig


def stream_dataset(
    name: str,
    hf_path: str,
    hf_subset: str | None = None,
    *,
    split: str = "train",
) -> Iterator[Any]:
    """Yield raw rows from a HF dataset in streaming mode.

    ``name`` is used only for debugging/logging context.
    """
    from datasets import load_dataset

    ds = load_dataset(
        hf_path,
        hf_subset,
        split=split,
        streaming=True,
    )
    for row in ds:
        yield row


def stream_validation(data_cfg: DataConfig) -> Iterator[Any]:
    """Yield deterministic validation examples.

    Token-budget stopping is applied in the tokenization/eval stage where
    token counts are available.
    """
    from datasets import load_dataset

    ds = load_dataset(
        data_cfg.validation.hf_path,
        data_cfg.validation.hf_subset,
        split="train",
        streaming=data_cfg.streaming,
    )
    # Deterministic stream order for comparable LM loss.
    if data_cfg.streaming:
        ds = ds.shuffle(seed=data_cfg.validation.seed, buffer_size=data_cfg.shuffle_buffer)
    for row in ds:
        yield row
