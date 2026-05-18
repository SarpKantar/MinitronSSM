"""Forward-hook based activation collection.

Reference: plan section 6.2.

The collector is used as a context manager::

    with ActivationCollector(model, targets=[...]) as col:
        for batch in calibration_loader:
            model(batch)
    scores = col.aggregate()

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import torch.nn as nn


class ActivationCollector:
    """Register forward hooks on selected modules and aggregate activations.

    This collector keeps only a reduced per-module vector (last-dimension
    profile), so it is memory-safe for long calibration runs.
    """

    def __init__(self, model: "nn.Module", targets: List[str]):
        self.model = model
        self.targets = targets
        self._handles: List[Any] = []
        self._buffers: Dict[str, Any] = {}
        self._counts: Dict[str, int] = {}

    @staticmethod
    def _to_tensor(output: Any):
        import torch

        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output:
            for x in output:
                if isinstance(x, torch.Tensor):
                    return x
        return None

    @staticmethod
    def _profile_last_dim(tensor) -> Any:
        import torch

        t = tensor.detach().to(torch.float32)
        t = t.abs()
        if t.ndim > 1:
            dims = tuple(range(t.ndim - 1))
            t = t.mean(dim=dims)
        return t.cpu()

    def __enter__(self) -> "ActivationCollector":
        modules = dict(self.model.named_modules())
        for name in self.targets:
            module = modules.get(name)
            if module is None:
                continue

            def _hook(_module, _inputs, outputs, mname=name):
                prof = self._to_tensor(outputs)
                if prof is None:
                    return
                vec = self._profile_last_dim(prof)
                if mname not in self._buffers:
                    self._buffers[mname] = vec
                    self._counts[mname] = 1
                else:
                    if self._buffers[mname].shape != vec.shape:
                        return
                    self._buffers[mname] += vec
                    self._counts[mname] += 1

            self._handles.append(module.register_forward_hook(_hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def aggregate(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self._buffers.items():
            n = max(1, self._counts.get(k, 1))
            out[k] = v / n
        return out
