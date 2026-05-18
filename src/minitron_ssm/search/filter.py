"""Post-enumeration filters and dedup helpers.

:func:`enumerate_candidates` already applies the budget filter; the
helpers here let scripts re-filter or sort an existing list (e.g.
after loading candidates from disk).
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from .space import Candidate


def filter_by_budget(
    candidates: Iterable[Candidate],
    target: int,
    tolerance: float,
) -> List[Candidate]:
    return [
        c for c in candidates
        if abs(c.param_count - target) <= tolerance * target
    ]


def dedup(candidates: Iterable[Candidate]) -> List[Candidate]:
    seen: set[Tuple[int, int, int, int, int]] = set()
    out: List[Candidate] = []
    for c in candidates:
        key = (c.layers, c.embedding, c.ffn, c.mamba_heads, c.mamba_head_channels)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def sort_by_distance_to_budget(
    candidates: Iterable[Candidate], target: int
) -> List[Candidate]:
    return sorted(candidates, key=lambda c: abs(c.param_count - target))
