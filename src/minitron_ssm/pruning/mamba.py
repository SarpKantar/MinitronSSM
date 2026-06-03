"""Mamba head + head-channel pruning.

Reference: plan section 7 ("Mamba Head Pruning",
"Mamba Head-Channel Pruning").

Affected tensors per Mamba layer (verify names with
:func:`models.inspect.describe_mamba_layout`):

* ``in_proj.weight`` rows for ``z``, ``x``, ``B``, ``C``, ``dt``
* ``conv1d.weight`` / ``conv1d.bias`` channels associated with x/B/C
* ``A_log``, ``D``, ``dt_bias`` (per-head)
* ``norm.weight`` (per-channel within d_inner)
* ``out_proj.weight`` input columns

TODO(stage-3): real implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch.nn as nn


def _get_int_attr(layer: "nn.Module", names: List[str], default: int) -> int:
    for n in names:
        if hasattr(layer, n):
            return int(getattr(layer, n))
    cfg = getattr(layer, "config", None)
    if cfg is not None:
        for n in names:
            if hasattr(cfg, n):
                return int(getattr(cfg, n))
    args = getattr(layer, "args", None)
    if args is not None:
        for n in names:
            if hasattr(args, n):
                return int(getattr(args, n))
    return int(default)


def _set_int_attr(layer: "nn.Module", names: List[str], value: int) -> None:
    for n in names:
        if hasattr(layer, n):
            setattr(layer, n, int(value))
    cfg = getattr(layer, "config", None)
    if cfg is not None:
        for n in names:
            if hasattr(cfg, n):
                setattr(cfg, n, int(value))
    args = getattr(layer, "args", None)
    if args is not None:
        for n in names:
            if hasattr(args, n):
                setattr(args, n, int(value))


def _slice_param_dim(param, dim: int, idx: List[int]):
    import torch
    import torch.nn as nn

    index = torch.as_tensor(idx, dtype=torch.long, device=param.device)
    new = torch.index_select(param.data, dim=dim, index=index).contiguous()
    return nn.Parameter(new, requires_grad=param.requires_grad)


def prune_mamba_heads(layer: "nn.Module", keep_idx_per_group: List[List[int]]) -> None:
    """Drop heads, keeping ``keep_idx_per_group[g]`` inside group ``g``.

    *In-place* mutation of the layer's parameter tensors. The number
    of kept heads must be the same across all groups (group-aware
    constraint, plan section 6.3).

    Best-effort implementation that slices any parameter dimension
    matching ``n_heads`` or ``d_inner`` (`n_heads * head_dim`).
    """
    if not keep_idx_per_group:
        raise ValueError("keep_idx_per_group must be non-empty")

    n_heads = _get_int_attr(layer, ["n_heads", "num_heads"], 0)
    n_groups = _get_int_attr(layer, ["n_groups", "num_groups"], 1)
    head_dim = _get_int_attr(layer, ["head_dim", "headdim", "d_head"], 0)
    if n_heads <= 0 or head_dim <= 0 or n_groups <= 0:
        return
    if n_heads % n_groups != 0:
        return

    heads_per_group = n_heads // n_groups
    keep_heads: List[int] = []
    for g, local_idx in enumerate(keep_idx_per_group):
        for i in local_idx:
            ii = int(i)
            if 0 <= ii < heads_per_group:
                keep_heads.append(g * heads_per_group + ii)
    keep_heads = sorted(set(keep_heads))
    if not keep_heads:
        raise ValueError("No valid head indices selected for pruning")

    keep_dinner: List[int] = []
    for h in keep_heads:
        base = h * head_dim
        keep_dinner.extend(base + c for c in range(head_dim))

    import torch.nn as nn

    old_dinner = n_heads * head_dim
    for mod in layer.modules():
        for pname, p in list(mod.named_parameters(recurse=False)):
            if p.ndim == 0:
                continue
            if p.shape[0] == n_heads:
                setattr(mod, pname, _slice_param_dim(p, 0, keep_heads))
                continue
            if p.ndim > 1 and p.shape[1] == n_heads:
                setattr(mod, pname, _slice_param_dim(p, 1, keep_heads))
                continue

            if p.shape[0] == old_dinner:
                setattr(mod, pname, _slice_param_dim(p, 0, keep_dinner))
                continue
            if p.ndim > 1 and p.shape[1] == old_dinner:
                setattr(mod, pname, _slice_param_dim(p, 1, keep_dinner))
                continue

            # Stacked projections where the leading dim is a multiple of d_inner.
            if p.shape[0] > old_dinner and p.shape[0] % old_dinner == 0:
                chunks = p.shape[0] // old_dinner
                idx = []
                for ch in range(chunks):
                    base = ch * old_dinner
                    idx.extend(base + i for i in keep_dinner)
                setattr(mod, pname, _slice_param_dim(p, 0, idx))

    _set_int_attr(layer, ["n_heads", "num_heads"], len(keep_heads))
    _set_int_attr(layer, ["d_inner", "inner_dim"], len(keep_heads) * head_dim)


def prune_mamba_head_channels(layer: "nn.Module", keep_channels: List[int]) -> None:
    """Drop head channels uniformly across all heads in *layer*.

    Best-effort implementation that slices `d_inner` dimensions while
    preserving head count.
    """
    if not keep_channels:
        raise ValueError("keep_channels must be non-empty")

    n_heads = _get_int_attr(layer, ["n_heads", "num_heads"], 0)
    head_dim = _get_int_attr(layer, ["head_dim", "headdim", "d_head"], 0)
    if n_heads <= 0 or head_dim <= 0:
        return

    keep_c = sorted(set(int(i) for i in keep_channels))
    if max(keep_c) >= head_dim:
        raise ValueError(
            f"channel index {max(keep_c)} out of bounds for head_dim={head_dim}"
        )

    keep_dinner: List[int] = []
    for h in range(n_heads):
        base = h * head_dim
        keep_dinner.extend(base + c for c in keep_c)

    import torch.nn as nn

    old_dinner = n_heads * head_dim
    for mod in layer.modules():
        for pname, p in list(mod.named_parameters(recurse=False)):
            if p.ndim == 0:
                continue
            if p.shape[0] == old_dinner:
                setattr(mod, pname, _slice_param_dim(p, 0, keep_dinner))
                continue
            if p.ndim > 1 and p.shape[1] == old_dinner:
                setattr(mod, pname, _slice_param_dim(p, 1, keep_dinner))
                continue

            if p.shape[0] > old_dinner and p.shape[0] % old_dinner == 0:
                chunks = p.shape[0] // old_dinner
                idx = []
                for ch in range(chunks):
                    base = ch * old_dinner
                    idx.extend(base + i for i in keep_dinner)
                setattr(mod, pname, _slice_param_dim(p, 0, idx))

    _set_int_attr(layer, ["head_dim", "headdim", "d_head"], len(keep_c))
    _set_int_attr(layer, ["d_inner", "inner_dim"], n_heads * len(keep_c))
