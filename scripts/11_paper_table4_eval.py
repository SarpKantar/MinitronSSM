"""Paper-aligned Table 4 evaluation for one model.

The paper mixes zero-shot and few-shot tasks, so this script evaluates each
shot group separately and checkpoints the output after every completed group.
It supports the parent, NVIDIA's released 4B reference, and local pruned/KD
candidate checkpoints reconstructed from their candidate metadata.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from _common import REPO_ROOT, base_parser, print_resolved

from minitron_ssm.models.load import (
    build_pruned_model,
    disable_fast_mamba_kernels,
    load_parent,
    load_reference_4b,
)
from minitron_ssm.utils.checkpoint import load_candidate
from minitron_ssm.utils.config import load_base
from minitron_ssm.utils.logging import get_logger

log = get_logger("paper-table4-eval")

SOCIAL_IQA_TASK: dict[str, Any] = {
    "task": "social_iqa",
    "dataset_path": "allenai/social_i_qa",
    # datasets>=4 no longer executes dataset scripts. Hugging Face's generated
    # parquet revision has the same fields and splits used by the harness task.
    "dataset_kwargs": {"revision": "refs/convert/parquet"},
    "output_type": "multiple_choice",
    "training_split": "train",
    "validation_split": "validation",
    "doc_to_text": "Q: {{context}} {{question}}\nA:",
    "target_delimiter": " ",
    "doc_to_choice": "{{[answerA, answerB, answerC]}}",
    "doc_to_target": "{{ (label|int) - 1 }}",
    "metric_list": [
        {
            "metric": "acc",
            "aggregation": "mean",
            "higher_is_better": True,
        }
    ],
    "metadata": {"version": 0.0},
}


PAPER_PROTOCOL: tuple[dict[str, Any], ...] = (
    {
        "name": "zero_shot",
        "num_fewshot": 0,
        "tasks": [
            "arc_challenge",
            "arc_easy",
            "commonsense_qa",
            "hellaswag",
            "openbookqa",
            "piqa",
            "race",
            "social_iqa",
            "truthfulqa_mc2",
            "winogrande",
        ],
        "unsafe_code": False,
    },
    {
        "name": "mmlu_5shot",
        "num_fewshot": 5,
        "tasks": ["mmlu"],
        "unsafe_code": False,
    },
    {
        "name": "gsm8k_8shot",
        "num_fewshot": 8,
        "tasks": ["gsm8k"],
        "unsafe_code": False,
    },
    {
        "name": "humaneval_0shot",
        "num_fewshot": 0,
        "tasks": ["humaneval"],
        "unsafe_code": True,
    },
    {
        "name": "mbpp_3shot",
        "num_fewshot": 3,
        "tasks": ["mbpp"],
        "unsafe_code": True,
    },
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish JSON through node-local storage with retries for shared FS errors."""
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "minitron_table4_json"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dir / f"{path.name}.{os.getpid()}.json"
    staged.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    last_error: OSError | None = None
    try:
        for attempt in range(8):
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                shutil.copyfile(staged, tmp)
                os.replace(tmp, path)
                return
            except OSError as exc:
                last_error = exc
                tmp.unlink(missing_ok=True)
                time.sleep(attempt + 1)
    finally:
        staged.unlink(missing_ok=True)
    raise OSError(f"failed to publish evaluation output: {path}") from last_error


def _read_existing(path: Path) -> dict[str, Any] | None:
    path = _resolve(path)
    if not path.exists():
        return None
    last_error: BaseException | None = None
    for attempt in range(8):
        try:
            with path.open("rb") as handle:
                data = json.loads(handle.read().decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{path} must contain a JSON object")
            return data
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(attempt + 1)
    raise OSError(f"failed to read existing evaluation output: {path}") from last_error


def _compact(result: Any) -> dict[str, Any]:
    compact = dict(result)
    compact.pop("samples", None)
    return compact


def _task_specs(task_names: list[str]) -> list[Any]:
    return [
        SOCIAL_IQA_TASK if task_name == "social_iqa" else task_name
        for task_name in task_names
    ]


def _load_model(args: Any, base: Any) -> tuple[Any, Any, str]:
    import torch

    if args.target == "parent":
        model, tokenizer = load_parent(base, eval_mode=True)
        checkpoint = base.models.parent
    elif args.target == "reference":
        model, tokenizer = load_reference_4b(base)
        checkpoint = base.models.reference_4b
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --target=checkpoint")
        checkpoint_path = _resolve(args.checkpoint)
        parent, tokenizer = load_parent(
            base,
            eval_mode=True,
            template_on_cpu=True,
        )
        state_dict, candidate_cfg = load_candidate(checkpoint_path)
        model = build_pruned_model(parent, candidate_cfg, eval_mode=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        del parent
        checkpoint = str(checkpoint_path)

    patched = disable_fast_mamba_kernels(model)
    if patched:
        log.info("disabled fused Mamba kernels for %d module(s)", patched)
    return model.eval(), tokenizer, str(checkpoint)


def _evaluate_group(
    model: Any,
    tokenizer: Any,
    group: dict[str, Any],
    *,
    batch_size: str,
    limit: float | None,
) -> dict[str, Any]:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        trust_remote_code=True,
    )
    result = simple_evaluate(
        model=lm,
        tasks=_task_specs(list(group["tasks"])),
        num_fewshot=int(group["num_fewshot"]),
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
        confirm_run_unsafe_code=bool(group["unsafe_code"]),
        random_seed=1234,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    if result is None:
        raise RuntimeError(f"lm-eval returned no result for group {group['name']}")
    return _compact(result)


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument(
        "--target",
        choices=("parent", "reference", "checkpoint"),
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Candidate checkpoint directory for --target=checkpoint.",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        default="1",
        help="Fixed lm-eval batch size; 1 is safest for Nemotron-H.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional lm-eval sample limit for smoke testing.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore completed groups in an existing output and rerun all groups.",
    )
    args = parser.parse_args()

    base = load_base(args.base_config)
    output_path = _resolve(args.output)
    checkpoint_arg = str(_resolve(args.checkpoint)) if args.checkpoint else None
    planned = {
        "target": args.target,
        "label": args.label,
        "checkpoint": checkpoint_arg,
        "output": str(output_path),
        "batch_size": args.batch_size,
        "limit": args.limit,
        "protocol": list(PAPER_PROTOCOL),
    }
    if args.dry_run:
        print_resolved("base", base)
        print(json.dumps(planned, indent=2))
        return 0

    previous = None if args.restart else _read_existing(output_path)
    payload: dict[str, Any] = previous or {
        "schema_version": 1,
        "paper": "Minitron-SSM",
        "paper_table": 4,
        "model": args.label,
        "target": args.target,
        "requested_checkpoint": checkpoint_arg,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "status": "running",
        "groups": {},
    }
    payload["status"] = "running"
    payload["last_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json_atomic(output_path, payload)

    model, tokenizer, resolved_checkpoint = _load_model(args, base)
    payload["checkpoint"] = resolved_checkpoint
    _write_json_atomic(output_path, payload)

    for group in PAPER_PROTOCOL:
        name = str(group["name"])
        existing = payload["groups"].get(name)
        if existing and existing.get("status") == "completed" and not args.restart:
            log.info("skipping completed group %s", name)
            continue

        log.info(
            "evaluating %s: tasks=%s fewshot=%d",
            name,
            ",".join(group["tasks"]),
            group["num_fewshot"],
        )
        payload["groups"][name] = {
            "status": "running",
            "tasks": list(group["tasks"]),
            "num_fewshot": int(group["num_fewshot"]),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _write_json_atomic(output_path, payload)
        try:
            result = _evaluate_group(
                model,
                tokenizer,
                group,
                batch_size=args.batch_size,
                limit=args.limit,
            )
        except BaseException as exc:
            payload["status"] = "failed"
            payload["groups"][name].update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
            _write_json_atomic(output_path, payload)
            raise

        payload["groups"][name].update(
            {
                "status": "completed",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "harness": result,
            }
        )
        _write_json_atomic(output_path, payload)
        log.info("completed group %s", name)

    payload["status"] = "completed"
    payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json_atomic(output_path, payload)
    log.info("wrote completed Table 4 evaluation to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
