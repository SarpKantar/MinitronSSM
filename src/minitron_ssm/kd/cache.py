"""Top-k teacher-logit cache (KD memory fallback path).

Reference: plan section 9.3 Risk 2.

The cache is built by running the teacher once over the KD dataset
and storing, per token position, the top-k ``(values, indices)`` of
the teacher's logits. Training then becomes a single-model
(student-only) loop driven by ``kd_topk_kl_loss``.

TODO(stage-7/8): real implementation.
"""

from __future__ import annotations

import json
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

    Uses torch shard files and a lightweight manifest.json.
    """

    def __init__(self, path: Path, k: int):
        self.path = path
        self.k = k

    def build(self, teacher: Any, loader: Iterator[Any]) -> None:
        import torch

        teacher.eval()
        self.path.mkdir(parents=True, exist_ok=True)
        device = next(teacher.parameters()).device

        shard_size = 256
        shard_idx = 0
        current: list[dict[str, torch.Tensor]] = []
        total_examples = 0
        shards: list[str] = []

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, dict) or "input_ids" not in batch:
                    continue
                input_ids = batch["input_ids"].to(device)
                out = teacher(input_ids=input_ids, use_cache=False)
                logits = out.logits
                vals, idx = torch.topk(logits, k=self.k, dim=-1)
                labels = batch.get("labels", batch["input_ids"])

                vals = vals.detach().to(torch.bfloat16).cpu()
                idx = idx.detach().to(torch.int32).cpu()
                labels = labels.detach().to(torch.long).cpu()

                bsz = vals.size(0)
                for i in range(bsz):
                    current.append(
                        {
                            "values": vals[i],
                            "indices": idx[i],
                            "tokens": labels[i],
                        }
                    )
                total_examples += bsz

                if len(current) >= shard_size:
                    shard_name = f"shard-{shard_idx:05d}.pt"
                    torch.save(current, self.path / shard_name)
                    shards.append(shard_name)
                    current = []
                    shard_idx += 1

        if current:
            shard_name = f"shard-{shard_idx:05d}.pt"
            torch.save(current, self.path / shard_name)
            shards.append(shard_name)

        manifest = {
            "k": self.k,
            "num_examples": total_examples,
            "shards": shards,
        }
        (self.path / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def iter_batches(self, batch_size: int, seq_len: int) -> Iterator[TopKEntry]:
        import torch

        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")

        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing cache manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        bucket: list[dict[str, torch.Tensor]] = []
        for shard_name in manifest.get("shards", []):
            shard_path = self.path / shard_name
            entries = torch.load(shard_path, map_location="cpu")
            for entry in entries:
                values = entry["values"][:seq_len]
                indices = entry["indices"][:seq_len].to(torch.long)
                tokens = entry["tokens"][:seq_len]
                if values.size(0) < seq_len:
                    pad = seq_len - values.size(0)
                    values = torch.nn.functional.pad(values, (0, 0, 0, pad))
                    indices = torch.nn.functional.pad(indices, (0, 0, 0, pad))
                    tokens = torch.nn.functional.pad(tokens, (0, pad), value=-100)
                bucket.append(
                    {
                        "values": values,
                        "indices": indices,
                        "tokens": tokens,
                    }
                )
                if len(bucket) == batch_size:
                    yield TopKEntry(
                        values=torch.stack([x["values"] for x in bucket], dim=0),
                        indices=torch.stack([x["indices"] for x in bucket], dim=0),
                        tokens=torch.stack([x["tokens"] for x in bucket], dim=0),
                    )
                    bucket = []

        if bucket:
            yield TopKEntry(
                values=torch.stack([x["values"] for x in bucket], dim=0),
                indices=torch.stack([x["indices"] for x in bucket], dim=0),
                tokens=torch.stack([x["tokens"] for x in bucket], dim=0),
            )
