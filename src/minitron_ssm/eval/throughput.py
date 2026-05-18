"""Throughput / latency / TTFT measurement.

Reference: plan sections 5.2, 11.

TODO(stage-1 / stage-5): real implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class ThroughputResult:
    input_len: int
    output_len: int
    batch_size: int
    tokens_per_second: float
    time_to_first_token_ms: float
    peak_memory_gib: float


def measure_throughput(
    model: "nn.Module",
    tokenizer,
    *,
    input_lens: List[int],
    output_lens: List[int],
    batch_sizes: List[int],
    warmup_iters: int = 3,
    measure_iters: int = 10,
) -> List[ThroughputResult]:
    """Measure throughput/latency across the Cartesian product of inputs.

    Uses synthetic prompts of repeated EOS/BOS IDs for repeatability.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

    results: List[ThroughputResult] = []

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for input_len in input_lens:
        for output_len in output_lens:
            for batch_size in batch_sizes:
                prompt = torch.full(
                    (batch_size, input_len), pad_id, dtype=torch.long, device=device
                )
                attn = torch.ones_like(prompt, device=device)

                # Warmup.
                with torch.no_grad():
                    for _ in range(max(0, warmup_iters)):
                        _ = model.generate(
                            input_ids=prompt,
                            attention_mask=attn,
                            max_new_tokens=output_len,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=pad_id,
                        )

                # Time-to-first-token: generate one token.
                _sync()
                start_ttft = time.perf_counter()
                with torch.no_grad():
                    _ = model.generate(
                        input_ids=prompt,
                        attention_mask=attn,
                        max_new_tokens=1,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=pad_id,
                    )
                _sync()
                ttft_ms = (time.perf_counter() - start_ttft) * 1000.0

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                # Throughput with full decode length.
                _sync()
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(max(1, measure_iters)):
                        _ = model.generate(
                            input_ids=prompt,
                            attention_mask=attn,
                            max_new_tokens=output_len,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=pad_id,
                        )
                _sync()
                elapsed = time.perf_counter() - start
                gen_tokens = batch_size * output_len * max(1, measure_iters)
                tps = gen_tokens / max(elapsed, 1e-8)

                peak_gib = 0.0
                if device.type == "cuda":
                    peak_gib = float(torch.cuda.max_memory_allocated(device)) / (1024**3)

                results.append(
                    ThroughputResult(
                        input_len=input_len,
                        output_len=output_len,
                        batch_size=batch_size,
                        tokens_per_second=float(tps),
                        time_to_first_token_ms=float(ttft_ms),
                        peak_memory_gib=float(peak_gib),
                    )
                )

    return results
