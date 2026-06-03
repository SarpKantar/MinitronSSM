"""Checkpoint save/load stubs.

Real implementations land alongside the pruning / KD stages. For now
these helpers fix the I/O contract so scripts can wire artifacts
together without importing torch on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch


def save_candidate(
    pruned_state_dict: "dict[str, Any]",
    candidate_config: "dict[str, Any]",
    out_dir: Path,
    candidate_id: str,
) -> Path:
    """Persist a pruned candidate.

    Layout::

        out_dir/<candidate_id>/
            config.json
            model.safetensors   # or pytorch_model.bin

    The function prefers ``model.safetensors`` when safetensors is
    available and falls back to ``pytorch_model.bin`` otherwise.
    """
    import torch

    out_path = out_dir / candidate_id
    out_path.mkdir(parents=True, exist_ok=True)

    cfg_path = out_path / "config.json"
    cfg_path.write_text(json.dumps(candidate_config, indent=2), encoding="utf-8")

    cpu_state = {
        k: (v.detach().cpu() if hasattr(v, "detach") else v)
        for k, v in pruned_state_dict.items()
    }
    safetensor_path = out_path / "model.safetensors"
    try:
        from safetensors.torch import save_file

        save_file(cpu_state, str(safetensor_path))
    except Exception:
        torch.save(cpu_state, out_path / "pytorch_model.bin")
    return out_path


def load_candidate(path: Path) -> "tuple[Any, dict[str, Any]]":
    """Load a candidate written by :func:`save_candidate`.

    Returns ``(state_dict, config_dict)`` where ``state_dict`` is a
    plain tensor dictionary that can be loaded into a model instance.
    """
    import torch

    if not path.exists():
        raise FileNotFoundError(f"candidate path not found: {path}")
    cfg_path = path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config.json under {path}")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    safetensor_path = path / "model.safetensors"
    if safetensor_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensor_path), device="cpu")
        return state_dict, config

    bin_path = path / "pytorch_model.bin"
    if bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
        return state_dict, config

    raise FileNotFoundError(f"no model weights found under {path}")


def cuda_memory_summary(device: Optional[int] = None) -> str:
    """Return a one-line GPU memory summary for the given device.

    Returns the empty string on CPU-only machines.
    """
    try:
        import torch  # local import keeps the module importable without torch

        if not torch.cuda.is_available():
            return ""
        idx = device if device is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(idx)
        used = total - free
        return (
            f"cuda:{idx} used={used / 1024**3:.2f}GiB "
            f"total={total / 1024**3:.2f}GiB"
        )
    except Exception:
        return ""
