"""Candidate-architecture enumeration.

Given a :class:`SearchSpaceConfig` (parsed from
``configs/search_space.yaml``) and the parent :class:`ArchSpec`,
enumerate up to ``max_candidates`` width-pruned configurations to be
evaluated downstream.

The enumeration policy is intentionally simple:

1. Anchors from the YAML are always emitted first.
2. We sweep the Cartesian product of the grids, keeping only
   configurations whose analytical parameter count lies inside the
   configured budget tolerance.
3. We pick candidates that are roughly evenly spread across the
   filtered set (deterministic stride) to avoid clustering.

This is *not* a random search; it is a deterministic enumeration so
that re-running stage 3 produces the same candidates.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List

from ..utils.config import SearchSpaceConfig
from .param_count import ArchSpec, LayerCounts, count_params

REPORTED_PARENT_PARAMS = 8_000_000_000


@dataclass(frozen=True)
class Candidate:
    id: str
    layers: int
    embedding: int
    ffn: int
    mamba_heads: int
    mamba_head_channels: int
    mamba_groups: int
    vocab_size: int
    param_count: int
    layer_counts: LayerCounts

    def to_arch(self) -> ArchSpec:
        return ArchSpec(
            layers=self.layers,
            embedding=self.embedding,
            ffn=self.ffn,
            mamba_heads=self.mamba_heads,
            mamba_head_channels=self.mamba_head_channels,
            mamba_groups=self.mamba_groups,
            vocab_size=self.vocab_size,
            layer_counts=self.layer_counts,
        )


def _layer_counts_for(parent: ArchSpec, layers: int) -> LayerCounts:
    """Scale parent layer-type ratios down to a target depth.

    Used for the depth-ablation branch; width-only candidates inherit
    the parent's exact counts.
    """
    if layers == parent.layers:
        return parent.layer_counts

    # Preserve the attention count (paper does not prune attention).
    attn = min(parent.layer_counts.attention, layers)
    remaining = layers - attn
    mamba_ratio = parent.layer_counts.mamba / max(
        parent.layer_counts.mamba + parent.layer_counts.mlp, 1
    )
    mamba = int(round(remaining * mamba_ratio))
    mlp = remaining - mamba
    return LayerCounts(attention=attn, mamba=mamba, mlp=mlp)


def _within_budget(n: int, target: int, tol: float) -> bool:
    return abs(n - target) <= tol * target


def enumerate_candidates(
    cfg: SearchSpaceConfig, parent: ArchSpec
) -> List[Candidate]:
    """Return up to ``cfg.max_candidates`` deterministic candidates."""
    out: List[Candidate] = []
    seen: set[tuple] = set()
    parent_raw_params = count_params(parent)
    scale = REPORTED_PARENT_PARAMS / parent_raw_params

    def _emit(
        cid: str,
        layers: int,
        emb: int,
        ffn: int,
        heads: int,
        head_ch: int,
    ) -> None:
        key = (layers, emb, ffn, heads, head_ch)
        if key in seen:
            return
        lc = _layer_counts_for(parent, layers)
        if lc.total != layers:
            return
        arch = ArchSpec(
            layers=layers,
            embedding=emb,
            ffn=ffn,
            mamba_heads=heads,
            mamba_head_channels=head_ch,
            mamba_groups=parent.mamba_groups,
            vocab_size=parent.vocab_size,
            layer_counts=lc,
        )
        if heads % parent.mamba_groups != 0:
            return  # group-aware constraint, plan section 6.3
        n_raw = count_params(arch)
        n = int(round(n_raw * scale))
        if not _within_budget(n, cfg.target_param_budget, cfg.budget_tolerance):
            return
        seen.add(key)
        out.append(
            Candidate(
                id=cid,
                layers=layers,
                embedding=emb,
                ffn=ffn,
                mamba_heads=heads,
                mamba_head_channels=head_ch,
                mamba_groups=parent.mamba_groups,
                vocab_size=parent.vocab_size,
                param_count=n,
                layer_counts=lc,
            )
        )

    # 1) Anchors first (always emitted, budget filter still applies).
    for a in cfg.anchors:
        _emit(a.id, a.layers, a.embedding, a.ffn, a.mamba_heads, a.mamba_head_channels)

    # 2) Cartesian sweep of width grids (and optional depth ablation).
    layers_options = list(cfg.grids.layers)
    if cfg.depth_ablation.enabled:
        layers_options = list(
            dict.fromkeys(layers_options + cfg.depth_ablation.layers_options)
        )

    grid = list(
        itertools.product(
            layers_options,
            cfg.grids.embedding,
            cfg.grids.ffn,
            cfg.grids.mamba_heads,
            cfg.grids.mamba_head_channels,
        )
    )

    # Deterministic, evenly spaced selection from the filtered grid.
    valid: List[tuple] = []
    for layers, emb, ffn, heads, head_ch in grid:
        lc = _layer_counts_for(parent, layers)
        if lc.total != layers:
            continue
        if heads % parent.mamba_groups != 0:
            continue
        arch = ArchSpec(
            layers=layers,
            embedding=emb,
            ffn=ffn,
            mamba_heads=heads,
            mamba_head_channels=head_ch,
            mamba_groups=parent.mamba_groups,
            vocab_size=parent.vocab_size,
            layer_counts=lc,
        )
        n_raw = count_params(arch)
        n = int(round(n_raw * scale))
        if _within_budget(n, cfg.target_param_budget, cfg.budget_tolerance):
            valid.append((layers, emb, ffn, heads, head_ch, n))

    valid.sort(key=lambda t: (abs(t[-1] - cfg.target_param_budget), t[:-1]))

    remaining = max(cfg.max_candidates - len(out), 0)
    for i, (layers, emb, ffn, heads, head_ch, _) in enumerate(valid):
        if remaining <= 0:
            break
        _emit(f"cand-{i:03d}", layers, emb, ffn, heads, head_ch)
        remaining = cfg.max_candidates - len(out)

    return out[: cfg.max_candidates]
