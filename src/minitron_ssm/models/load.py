"""Load the Nemotron-H teacher / parent model.

Reference: plan sections 2 and 5.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from ..utils.config import BaseConfig

if TYPE_CHECKING:
    import torch


def _dtype_from_precision(precision: str):
    import torch

    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if precision not in mapping:
        raise ValueError(
            f"Unsupported precision={precision!r}; expected one of {sorted(mapping)}"
        )
    return mapping[precision]


def _assert_parent_arch_compat(model: Any, base_cfg: BaseConfig) -> None:
    """Validate major dimensions against ``base_cfg.parent_arch``.

    Some Nemotron-H internals are custom and field names can differ across
    releases; we fail only on fields that are actually present in the loaded
    config to avoid false positives.
    """
    cfg = model.config
    expect = base_cfg.parent_arch

    checks: list[tuple[str, int]] = [
        ("num_hidden_layers", expect.layers),
        ("hidden_size", expect.embedding),
        ("intermediate_size", expect.ffn),
        ("vocab_size", expect.vocab_size),
    ]
    for key, val in checks:
        got = getattr(cfg, key, None)
        if got is not None and got != val:
            raise ValueError(f"Model/config mismatch for {key}: got {got}, expected {val}")


def disable_fast_mamba_kernels(model: Any) -> int:
    """Force Nemotron-H Mamba blocks onto the safe torch path.

    The installed ``mamba_ssm`` / ``causal-conv1d`` stack can execute
    inference successfully but fail in the fused training kernel
    ``mamba_split_conv1d_scan_combined`` with
    ``TypeError: causal_conv1d_fwd(): incompatible function arguments``.
    Repointing ``cuda_kernels_forward`` to the model's own
    ``torch_forward`` keeps the code on GPU while bypassing the broken
    fused path.
    """
    patched = 0
    for module in model.modules():
        if hasattr(module, "cuda_kernels_forward") and hasattr(module, "torch_forward"):
            module.cuda_kernels_forward = module.torch_forward
            patched += 1
    return patched


def load_parent(
    base_cfg: BaseConfig,
    *,
    eval_mode: bool = True,
) -> Tuple[Any, Any]:
    """Load ``base_cfg.models.parent`` and its tokenizer.

    Returns
    -------
    (model, tokenizer)
        ``model`` in ``base_cfg.hardware.precision`` on the configured
        device; ``tokenizer`` is the matching HF tokenizer.

    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = base_cfg.paths.hf_cache
    torch_dtype = _dtype_from_precision(base_cfg.hardware.precision)

    tokenizer = AutoTokenizer.from_pretrained(
        base_cfg.models.parent,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available() and base_cfg.hardware.device == "cuda":
        # Let transformers shard across visible GPUs automatically.
        model = AutoModelForCausalLM.from_pretrained(
            base_cfg.models.parent,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_cfg.models.parent,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        model = model.to("cpu")

    if eval_mode:
        model.eval()

    _assert_parent_arch_compat(model, base_cfg)
    return model, tokenizer


def build_pruned_model(
    parent: Any,
    cand_cfg: "dict[str, Any]",
    *,
    eval_mode: bool = True,
) -> Any:
    """Instantiate a pruned model shell whose architecture matches *cand_cfg*.

    Instead of deepcopying the parent (which would duplicate 16 GB on GPU),
    we copy the parent's HuggingFace config, override the pruned dimensions,
    build a fresh (randomly initialised) model from that config, and return
    it ready for ``load_state_dict``.

    Mapping between :class:`~minitron_ssm.search.space.Candidate` fields and
    Nemotron-H HuggingFace config attributes::

        embedding  → hidden_size
        ffn        → intermediate_size
        layers     → num_hidden_layers
    """
    import copy as _copy

    import torch
    from transformers import AutoModelForCausalLM

    pruned_config = _copy.deepcopy(parent.config)

    # NOTE: Mamba head/channel/group dims are intentionally NOT overridden here.
    # The stage-04 pruning silently no-op'd on those dimensions (prune_mamba_heads
    # was called on the NemotronHBlock rather than its .mixer, so _get_int_attr
    # returned 0 and the early-return guard fired). The saved checkpoints therefore
    # retain the original Mamba tensor shapes (num_heads=128, head_dim=64). Only
    # override dims that were actually pruned so the model shell matches the ckpt.
    overrides: dict[str, tuple[str, ...]] = {
        "embedding": ("hidden_size",),
        "ffn": ("intermediate_size",),
        "layers": ("num_hidden_layers",),
    }
    for cand_key, cfg_attrs in overrides.items():
        if cand_key not in cand_cfg:
            continue
        for cfg_attr in cfg_attrs:
            if hasattr(pruned_config, cfg_attr):
                setattr(pruned_config, cfg_attr, cand_cfg[cand_key])

    model = AutoModelForCausalLM.from_config(pruned_config, trust_remote_code=True)
    model = model.to(dtype=parent.dtype if hasattr(parent, "dtype") else torch.bfloat16)
    if eval_mode:
        model.eval()
    return model


def load_reference_4b(base_cfg: BaseConfig) -> Tuple[Any, Any]:
    """Load NVIDIA's released 4B model as a reference-only upper bound.

    Used only by Stage 9 evaluation for context; never used as the teacher
    during KD in this reproduction pipeline.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = base_cfg.paths.hf_cache
    torch_dtype = _dtype_from_precision(base_cfg.hardware.precision)

    tokenizer = AutoTokenizer.from_pretrained(
        base_cfg.models.reference_4b,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_cfg.models.reference_4b,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    if torch.cuda.is_available() and base_cfg.hardware.device == "cuda":
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    model.eval()
    return model, tokenizer
