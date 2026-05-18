"""Checkpoint save/load stubs.

Real implementations land alongside the pruning / KD stages. For now
these helpers fix the I/O contract so scripts can wire artifacts
together without importing torch on CPU.
"""

from __future__ import annotations

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

    TODO(stage-4): implement with ``safetensors.torch.save_file`` and
    write the candidate config as JSON. Returns the directory path.
    """
    raise NotImplementedError("TODO(stage-4): implement candidate checkpoint write")


def load_candidate(path: Path) -> "tuple[Any, dict[str, Any]]":
    """Load a candidate written by :func:`save_candidate`.

    TODO(stage-5): returns ``(model, config_dict)``. Needs to handle
    both safetensors and pytorch_model.bin shards.
    """
    raise NotImplementedError("TODO(stage-5): implement candidate checkpoint read")


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
