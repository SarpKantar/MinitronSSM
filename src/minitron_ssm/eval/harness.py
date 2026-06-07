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
    tokenizer: Any = None,
    batch_size: Any = None,
) -> Dict[str, Any]:
    """Run the configured lm-eval-harness task list.

    Two modes:

    * ``model_or_path`` is a string / :class:`~pathlib.Path` -> evaluated via
      the built-in ``model="hf"`` path loader (``from_pretrained``). Use this
      for models stored as a proper HuggingFace checkpoint (e.g. the parent).
    * ``model_or_path`` is an in-memory ``transformers`` model -> wrapped in
      lm-eval's ``HFLM``. Use this for the pruned / KD students, whose on-disk
      ``config.json`` is candidate metadata (not an HF config) and therefore
      cannot be loaded with ``from_pretrained``. A matching ``tokenizer`` must
      be supplied in this mode.
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

    bs = batch_size if batch_size is not None else cfg.lm_eval_harness.batch_size
    num_fewshot = int(cfg.lm_eval_harness.num_fewshot)

    if isinstance(model_or_path, (str, Path)):
        out = simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_or_path},trust_remote_code=True",
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=bs,
        )
        return dict(out)

    if tokenizer is None:
        raise ValueError(
            "run_harness with an in-memory model requires a matching tokenizer"
        )

    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model_or_path, tokenizer=tokenizer, batch_size=bs)
    out = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=bs,
    )
    return dict(out)


def task_list(cfg: EvalConfig, include_optional: bool = False) -> List[str]:
    if include_optional:
        return list(cfg.lm_eval_harness.tasks) + list(cfg.lm_eval_harness.optional_tasks)
    return list(cfg.lm_eval_harness.tasks)
