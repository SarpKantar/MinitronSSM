"""Analytical parameter counter for Nemotron-H-style hybrid models.

Used by :mod:`minitron_ssm.search.filter` to pick candidates that fit
inside the ~4B parameter budget *before* we actually instantiate
anything on the GPU.

The formula is an approximation tuned to the documented Nemotron-H 8B
layout. Once the real model is loaded in stage 1, we calibrate the
counter by comparing against ``sum(p.numel() for p in model.parameters())``
and adjust the per-layer-type weights in this module (or in base.yaml)
if needed. The test in ``tests/test_param_count.py`` only requires
order-of-magnitude correctness for now.

Reference: plan section 8.

Block-type assumptions (verify against ``model.config`` later)
-------------------------------------------------------------
* Nemotron-H 8B has 52 "layers" total, of which 4 are attention.
* The remaining 48 layers alternate between Mamba2 and MLP/FFN blocks.
  We assume an even 24/24 split unless overridden via ``LayerCounts``.
* Mamba2 uses a single fused ``in_proj`` producing ``[z, x, B, C, dt]``.
* MLP is SwiGLU: a fused gate-up projection followed by down projection.
* Token embedding and LM head are untied (separate matrices).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerCounts:
    """How many of each block type the parent contains."""

    attention: int = 4
    mamba: int = 24
    mlp: int = 24

    @property
    def total(self) -> int:
        return self.attention + self.mamba + self.mlp


@dataclass(frozen=True)
class ArchSpec:
    """Inputs to the analytical counter."""

    layers: int                  # total = attention + mamba + mlp
    embedding: int
    ffn: int
    mamba_heads: int
    mamba_head_channels: int
    mamba_groups: int = 8
    mamba_d_state: int = 128     # canonical Mamba2 default
    mamba_d_conv: int = 4
    vocab_size: int = 131_072
    tied_embeddings: bool = False
    layer_counts: LayerCounts = LayerCounts()


def _embedding_params(arch: ArchSpec) -> int:
    embed = arch.vocab_size * arch.embedding
    if arch.tied_embeddings:
        return embed
    return 2 * embed  # embedding + lm_head


def _mamba_layer_params(arch: ArchSpec) -> int:
    h = arch.embedding
    n_heads = arch.mamba_heads
    d_head = arch.mamba_head_channels
    d_inner = n_heads * d_head
    n_groups = arch.mamba_groups
    d_state = arch.mamba_d_state
    d_conv = arch.mamba_d_conv

    # Fused in_proj: produces z (d_inner), x (d_inner), B (n_groups*d_state),
    # C (n_groups*d_state), dt (n_heads).
    in_out = 2 * d_inner + 2 * n_groups * d_state + n_heads
    in_proj = h * in_out

    # Depthwise conv over (x, B, C).
    conv_channels = d_inner + 2 * n_groups * d_state
    conv1d = conv_channels * d_conv + conv_channels  # weight + bias

    # SSM scalars.
    a_log = n_heads
    d_param = n_heads
    dt_bias = n_heads

    # GroupRMSNorm + out_proj.
    norm = d_inner
    out_proj = d_inner * h

    return in_proj + conv1d + a_log + d_param + dt_bias + norm + out_proj


def _attention_layer_params(arch: ArchSpec) -> int:
    h = arch.embedding
    # Standard MHA with output projection. The paper says attention is not
    # pruned and is only ~8% of layers, so an exact MHA form is sufficient.
    qkv = 3 * h * h
    o = h * h
    return qkv + o


def _mlp_layer_params(arch: ArchSpec) -> int:
    h = arch.embedding
    f = arch.ffn
    # SwiGLU: fused gate+up projection (2 * f rows) + down projection.
    return h * (2 * f) + f * h


def _norm_params(arch: ArchSpec) -> int:
    # One RMSNorm per layer + a final pre-LM-head RMSNorm.
    return (arch.layer_counts.total + 1) * arch.embedding


def count_params(arch: ArchSpec) -> int:
    """Return the analytical parameter count for *arch*."""
    if arch.layer_counts.total != arch.layers:
        raise ValueError(
            f"layer_counts.total={arch.layer_counts.total} does not match "
            f"arch.layers={arch.layers}"
        )

    total = _embedding_params(arch)
    total += arch.layer_counts.attention * _attention_layer_params(arch)
    total += arch.layer_counts.mamba * _mamba_layer_params(arch)
    total += arch.layer_counts.mlp * _mlp_layer_params(arch)
    total += _norm_params(arch)
    return total


def parent_arch_from_base(base_cfg) -> ArchSpec:
    """Build an :class:`ArchSpec` matching ``configs/base.yaml``.

    Splits the non-attention layers evenly between Mamba and MLP.
    """
    non_attn = base_cfg.parent_arch.layers - base_cfg.parent_arch.attention_layers
    if non_attn < 0:
        raise ValueError("attention_layers exceeds total layers")
    mamba = non_attn // 2
    mlp = non_attn - mamba
    return ArchSpec(
        layers=base_cfg.parent_arch.layers,
        embedding=base_cfg.parent_arch.embedding,
        ffn=base_cfg.parent_arch.ffn,
        mamba_heads=base_cfg.parent_arch.mamba_heads,
        mamba_head_channels=base_cfg.parent_arch.mamba_head_channels,
        mamba_groups=base_cfg.parent_arch.mamba_groups,
        vocab_size=base_cfg.parent_arch.vocab_size,
        layer_counts=LayerCounts(
            attention=base_cfg.parent_arch.attention_layers,
            mamba=mamba,
            mlp=mlp,
        ),
    )
