"""Tokenization + packing into fixed-length sequences.

Reference: plan section 5.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator


def packed_token_batches(
    text_stream: Iterable[Any],
    tokenizer: Any,
    seq_len: int,
    batch_size: int,
) -> Iterator[Any]:
    """Tokenize *text_stream* and yield ``(batch_size, seq_len)`` int tensors.

    Yields dicts with:
    - ``input_ids``: ``(batch_size, seq_len)``
    - ``labels``: ``(batch_size, seq_len)``
    where labels are a clone of input_ids (causal next-token shift happens
    inside model loss implementations).
    """
    import torch

    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer does not define eos_token_id")

    def _extract_text(row: Any) -> str:
        if isinstance(row, str):
            return row
        if isinstance(row, dict):
            for key in ("text", "content", "prompt", "document"):
                if key in row and isinstance(row[key], str):
                    return row[key]
        return str(row)

    token_buffer: list[int] = []
    sequence_buffer: list[torch.Tensor] = []

    for row in text_stream:
        text = _extract_text(row)
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        token_buffer.extend(token_ids)
        token_buffer.append(eos_id)

        while len(token_buffer) >= seq_len:
            seq = token_buffer[:seq_len]
            del token_buffer[:seq_len]
            sequence_buffer.append(torch.tensor(seq, dtype=torch.long))
            if len(sequence_buffer) >= batch_size:
                batch = torch.stack(sequence_buffer[:batch_size], dim=0)
                del sequence_buffer[:batch_size]
                yield {"input_ids": batch, "labels": batch.clone()}
