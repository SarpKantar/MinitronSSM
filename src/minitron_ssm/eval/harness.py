"""``lm-evaluation-harness`` wrapper.

Reference: plan section 11 (Internal Table 4 reproduction).

TODO(stage-9): real implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..utils.config import EvalConfig


def run_harness(
    model_or_path: Any,
    cfg: EvalConfig,
    *,
    include_optional: bool = False,
) -> Dict[str, Any]:
    """Run the configured lm-eval-harness task list.

    Currently supports path-based evaluation (`model="hf"`). In-memory
    model wrappers can be added later if needed.
    """
    tasks = task_list(cfg, include_optional=include_optional)
    try:
        from lm_eval import simple_evaluate
    except Exception as e:
        return {
            "status": "skipped",
            "reason": f"lm-eval-harness import failed: {e}",
            "tasks": tasks,
        }

    if not isinstance(model_or_path, (str, Path)):
        raise NotImplementedError(
            "run_harness currently expects a checkpoint path/string. "
            "In-memory model objects are not wired yet."
        )

    model_path = str(model_or_path)
    model_args = f"pretrained={model_path},trust_remote_code=True"
    out = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=int(cfg.lm_eval_harness.num_fewshot),
        batch_size=cfg.lm_eval_harness.batch_size,
    )
    return dict(out)


def task_list(cfg: EvalConfig, include_optional: bool = False) -> List[str]:
    if include_optional:
        return list(cfg.lm_eval_harness.tasks) + list(cfg.lm_eval_harness.optional_tasks)
    return list(cfg.lm_eval_harness.tasks)
