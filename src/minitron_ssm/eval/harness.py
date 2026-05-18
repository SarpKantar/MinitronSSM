"""``lm-evaluation-harness`` wrapper.

Reference: plan section 11 (Internal Table 4 reproduction).

TODO(stage-9): real implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..utils.config import EvalConfig


def run_harness(
    model_or_path: Any,
    cfg: EvalConfig,
    *,
    include_optional: bool = False,
) -> Dict[str, Any]:
    """Run the configured lm-eval-harness task list.

    TODO(stage-9):
        * Use ``lm_eval.simple_evaluate(model="hf", model_args=..., tasks=...)``
          when ``model_or_path`` is a HF checkpoint dir.
        * Use ``lm_eval.evaluator.evaluate`` with an in-memory wrapper
          when called with a loaded model.
        * Honour ``num_fewshot`` and ``batch_size`` from ``cfg``.
    """
    raise NotImplementedError("TODO(stage-9): wrap lm-evaluation-harness")


def task_list(cfg: EvalConfig, include_optional: bool = False) -> List[str]:
    if include_optional:
        return list(cfg.lm_eval_harness.tasks) + list(cfg.lm_eval_harness.optional_tasks)
    return list(cfg.lm_eval_harness.tasks)
