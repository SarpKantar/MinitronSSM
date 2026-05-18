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
