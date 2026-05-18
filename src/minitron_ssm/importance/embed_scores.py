"""Global hidden-dimension (embedding) importance scoring.

Reference: plan section 6.4.

Hidden-dim pruning is global across the model: a single keep-index
must be applied to every projection that consumes or produces the
hidden state. We therefore aggregate channel importance across all
layers that use the hidden dimension before selecting indices.

"""

from __future__ import annotations

from typing import Any, Dict, List


def score_hidden_channels(activations: Dict[str, Any]) -> Any:
    """Return a single tensor of shape ``(embedding,)``.
    """
    import torch

    vectors = []
    for _k, val in activations.items():
        t = torch.as_tensor(val, dtype=torch.float32)
        if t.ndim > 1:
            t = t.abs().mean(dim=tuple(range(t.ndim - 1)))
        else:
            t = t.abs()
        vectors.append(t.flatten())

    if not vectors:
        raise ValueError("No activation vectors were provided")

    # Align by most common hidden dimension and sum scores across layers.
    lens = [int(v.numel()) for v in vectors]
    target_dim = max(set(lens), key=lens.count)
    aligned = [v for v in vectors if v.numel() == target_dim]
    if not aligned:
        raise ValueError("Could not find a consistent hidden dimension in activations")
    stacked = torch.stack(aligned, dim=0)
    return stacked.sum(dim=0)


def select_top_hidden_channels(scores: Any, target_embedding: int) -> List[int]:
    """Return a sorted list of ``target_embedding`` channel indices to keep.

    """
    import torch

    if target_embedding <= 0:
        raise ValueError("target_embedding must be > 0")
    t = torch.as_tensor(scores, dtype=torch.float32).flatten()
    if target_embedding > t.numel():
        raise ValueError(
            f"target_embedding={target_embedding} exceeds available={t.numel()}"
        )
    idx = torch.topk(t, k=target_embedding, largest=True).indices.tolist()
    return sorted(int(i) for i in idx)
