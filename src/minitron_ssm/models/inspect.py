"""Introspect the loaded Nemotron-H model.

The group-aware Mamba pruning code (plan section 6.3) needs to know:

* the number of SSM groups,
* the number of Mamba heads per layer,
* the head-channel dimension,
* the parameter names of ``in_proj`` / ``out_proj`` / ``conv1d`` /
  ``A_log`` / ``D`` / ``dt_bias`` so we can slice them correctly,
* which layer indices are Mamba vs. MLP vs. attention.

This module must be written *after* the model has been loaded once
(stage 1) because the exact attribute names depend on the
``modeling_nemotronh.py`` shipped with the HF checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class MambaLayerSpec:
    """Per-layer Mamba metadata."""

    layer_idx: int
    n_heads: int
    head_channels: int
    n_groups: int
    d_state: int
    d_conv: int
    module_path: str
    # Parameter name -> dtype/shape, filled by describe_mamba_layout.
    param_shapes: Dict[str, "tuple[int, ...]"] = field(default_factory=dict)


@dataclass
class MambaLayout:
    """Whole-model Mamba layout."""

    mamba_layer_indices: List[int]
    mlp_layer_indices: List[int]
    attention_layer_indices: List[int]
    layers: List[MambaLayerSpec]


def _get_attr_any(obj: Any, names: List[str], default: Any) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    cfg = getattr(obj, "config", None)
    if cfg is not None:
        for n in names:
            if hasattr(cfg, n):
                return getattr(cfg, n)
    args = getattr(obj, "args", None)
    if args is not None:
        for n in names:
            if hasattr(args, n):
                return getattr(args, n)
    return default


def _locate_layer_container(model: "nn.Module") -> tuple[str, Any]:
    """Return ``(path, container)`` for the main layer list."""
    candidates = [
        "model.layers",
        "model.backbone.layers",
        "backbone.layers",
        "layers",
        "transformer.h",
        "gpt_neox.layers",
    ]
    for dotted in candidates:
        cur = model
        ok = True
        for part in dotted.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and hasattr(cur, "__len__"):
            return dotted, cur
    raise ValueError(
        "Could not locate layer container on model. Tried: " + ", ".join(candidates)
    )


def _layer_kind(layer: "nn.Module") -> str:
    lname = layer.__class__.__name__.lower()
    child_names = " ".join(name.lower() for name, _ in layer.named_modules())
    blob = f"{lname} {child_names}"
    if "mamba" in blob or "ssm" in blob:
        return "mamba"
    if "attn" in blob or "attention" in blob:
        return "attention"
    return "mlp"


def describe_mamba_layout(model: "nn.Module") -> MambaLayout:
    """Walk the model and return a :class:`MambaLayout`.

    Works on Nemotron-H and other hybrid models by using name-based
    heuristics instead of hard ``isinstance`` checks.
    """
    layer_path, layers = _locate_layer_container(model)

    mamba_idxs: List[int] = []
    mlp_idxs: List[int] = []
    attn_idxs: List[int] = []
    specs: List[MambaLayerSpec] = []

    for i, layer in enumerate(layers):
        kind = _layer_kind(layer)
        if kind == "mamba":
            mamba_idxs.append(i)
            n_heads = int(_get_attr_any(layer, ["n_heads", "num_heads"], 1))
            head_channels = int(
                _get_attr_any(layer, ["head_dim", "headdim", "d_head"], 1)
            )
            n_groups = int(_get_attr_any(layer, ["n_groups", "num_groups"], 1))
            d_state = int(_get_attr_any(layer, ["d_state"], 0))
            d_conv = int(_get_attr_any(layer, ["d_conv"], 0))

            param_shapes: Dict[str, tuple[int, ...]] = {}
            for pn, p in layer.named_parameters(recurse=True):
                pkey = pn.lower()
                if any(
                    token in pkey
                    for token in ("in_proj", "out_proj", "conv", "a_log", "dt", ".d")
                ):
                    param_shapes[f"{layer_path}.{i}.{pn}"] = tuple(p.shape)

            specs.append(
                MambaLayerSpec(
                    layer_idx=i,
                    n_heads=n_heads,
                    head_channels=head_channels,
                    n_groups=n_groups,
                    d_state=d_state,
                    d_conv=d_conv,
                    module_path=f"{layer_path}.{i}",
                    param_shapes=param_shapes,
                )
            )
        elif kind == "attention":
            attn_idxs.append(i)
        else:
            mlp_idxs.append(i)

    return MambaLayout(
        mamba_layer_indices=mamba_idxs,
        mlp_layer_indices=mlp_idxs,
        attention_layer_indices=attn_idxs,
        layers=specs,
    )


def find_ffn_params(model: "nn.Module") -> Dict[int, Dict[str, str]]:
    """Map ``layer_idx -> {"gate_up": name, "down": name}`` for each MLP layer.

    Uses name heuristics to map layer-specific FFN projections.
    """
    _, layers = _locate_layer_container(model)
    out: Dict[int, Dict[str, str]] = {}
    for i, layer in enumerate(layers):
        names = [n for n, _ in layer.named_parameters(recurse=True)]
        gate_up = next(
            (
                n
                for n in names
                if any(k in n.lower() for k in ("gate_up_proj", "up_proj", "fc1"))
            ),
            None,
        )
        down = next(
            (
                n
                for n in names
                if any(k in n.lower() for k in ("down_proj", "fc2", "proj_down"))
            ),
            None,
        )
        if gate_up and down:
            out[i] = {"gate_up": gate_up, "down": down}
    return out


def find_embedding_params(model: "nn.Module") -> Dict[str, str]:
    """Return the names of the token-embedding and LM-head parameters.

    Returns keys: ``embedding`` and ``lm_head`` when found.
    """
    embedding_name = ""
    lm_head_name = ""

    for name, _ in model.named_parameters():
        lname = name.lower()
        if not embedding_name and any(
            tok in lname for tok in ("embed_tokens.weight", "wte.weight", "embedding.weight")
        ):
            embedding_name = name
        if not lm_head_name and lname.endswith("lm_head.weight"):
            lm_head_name = name
        if embedding_name and lm_head_name:
            break

    out: Dict[str, str] = {}
    if embedding_name:
        out["embedding"] = embedding_name
    if lm_head_name:
        out["lm_head"] = lm_head_name
    return out
