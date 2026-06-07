"""Build compact, report-ready artifacts from completed reproduction outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from _common import REPO_ROOT


EVAL_DIR = REPO_ROOT / "outputs" / "eval"
CANDIDATE_DIR = REPO_ROOT / "outputs" / "candidates"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints"

TASKS = (
    ("arc_challenge", "acc_norm,none"),
    ("arc_easy", "acc_norm,none"),
    ("hellaswag", "acc_norm,none"),
    ("piqa", "acc_norm,none"),
    ("winogrande", "acc,none"),
    ("openbookqa", "acc_norm,none"),
    ("mmlu", "acc,none"),
)

SOURCE_FILES = (
    "outputs/candidates/candidates.json",
    "outputs/eval/01_baseline.json",
    "outputs/eval/05_candidates.json",
    "outputs/eval/06_top3.json",
    "outputs/eval/08_kd_results.json",
    "outputs/eval/08_kd_results_round2.json",
    "outputs/eval/09_final_eval.json",
    "outputs/eval/09_final_eval_mini_kd.json",
    "outputs/eval/09_final_eval_round2.json",
    "outputs/eval/10_mini_kd.json",
    "outputs/eval/10_mini_kd_round2.json",
    "outputs/eval/12_report_pre_kd.json",
    "outputs/eval/12_report_kd15m.json",
    "outputs/eval/12_report_final35m.json",
    "outputs/eval/12_report_reference4b_native.json",
    "outputs/eval/12_report_final35m_efficiency.json",
)

JOB_IDS = (
    "1280917",
    "1280919",
    "1285584",
    "1285594",
    "1286114",
    "1286115",
    "1286270",
    "1287432",
    "1287433",
    "1287577",
    "1287578",
    "1287579",
    "1287580",
    "1290826",
)

JOB_NOTES = {
    "1280919": "Timed out after stages 02-06 had produced valid artifacts.",
    "1285594_2": "Post-Python exit 6; cand-006 training artifact and checkpoint are valid.",
    "1286270_2": "Post-Python exit 6; cand-006 round-2 artifact and checkpoint are valid.",
    "1287577": "Cancelled after preserving completed zero-shot and 5-shot MMLU groups.",
    "1287578": "Official-4B mixed-shot run failed on 5-shot MMLU CUDA OOM; zero-shot group preserved.",
    "1287579": "Cancelled while pending; no compute allocation.",
    "1287580": "Cancelled after preserving completed zero-shot and 5-shot MMLU groups.",
}

EXECUTION_LINEAGE = (
    {
        "job_id": "1280917",
        "stage": "01",
        "checkpoints": [],
        "artifacts": ["outputs/eval/01_baseline.json"],
    },
    {
        "job_id": "1280919",
        "stage": "02-06",
        "checkpoints": [
            "outputs/checkpoints/anchor-2",
            *[f"outputs/checkpoints/cand-{index:03d}" for index in range(19)],
        ],
        "artifacts": [
            "outputs/scores/importance_scores.pt",
            "outputs/candidates/candidates.json",
            "outputs/eval/05_candidates.json",
            "outputs/eval/06_top3.json",
        ],
    },
    {
        "job_id": "1285584",
        "stage": "07",
        "checkpoints": ["outputs/checkpoints/cand-009-kd-smoke"],
        "artifacts": ["outputs/eval/07_kd_smoke.json"],
    },
    {
        "job_id": "1285594_0",
        "stage": "08",
        "checkpoints": ["outputs/checkpoints/cand-009-kd"],
        "artifacts": ["outputs/eval/08_kd_results.cand00.json"],
    },
    {
        "job_id": "1285594_1",
        "stage": "08",
        "checkpoints": ["outputs/checkpoints/cand-016-kd"],
        "artifacts": ["outputs/eval/08_kd_results.cand01.json"],
    },
    {
        "job_id": "1285594_2",
        "stage": "08",
        "checkpoints": ["outputs/checkpoints/cand-006-kd"],
        "artifacts": ["outputs/eval/08_kd_results.cand02.json"],
    },
    {
        "job_id": "1286114",
        "stage": "09",
        "checkpoints": [],
        "artifacts": ["outputs/eval/09_final_eval.json"],
    },
    {
        "job_id": "1286115",
        "stage": "10",
        "checkpoints": ["outputs/checkpoints/cand-016-mini-final"],
        "artifacts": ["outputs/eval/10_mini_kd.json"],
    },
    {
        "job_id": "1286270_0",
        "stage": "08b",
        "checkpoints": ["outputs/checkpoints/cand-009-kd2"],
        "artifacts": ["outputs/eval/08_kd_results_round2.cand00.json"],
    },
    {
        "job_id": "1286270_1",
        "stage": "08b",
        "checkpoints": ["outputs/checkpoints/cand-016-kd2"],
        "artifacts": ["outputs/eval/08_kd_results_round2.cand01.json"],
    },
    {
        "job_id": "1286270_2",
        "stage": "08b",
        "checkpoints": ["outputs/checkpoints/cand-006-kd2"],
        "artifacts": ["outputs/eval/08_kd_results_round2.cand02.json"],
    },
    {
        "job_id": "1287432",
        "stage": "10b",
        "checkpoints": ["outputs/checkpoints/cand-016-mini-final2"],
        "artifacts": ["outputs/eval/10_mini_kd_round2.json"],
    },
    {
        "job_id": "1287433_0",
        "stage": "09b",
        "checkpoints": [],
        "artifacts": ["outputs/eval/09_final_eval_round2.json"],
    },
    {
        "job_id": "1287433_1",
        "stage": "09c",
        "checkpoints": [],
        "artifacts": ["outputs/eval/09_final_eval_mini_kd.json"],
    },
    {
        "job_id": "1287577",
        "stage": "11-parent-partial",
        "checkpoints": [],
        "artifacts": [
            "outputs/eval/11_table4_parent.partial-preserved-20260606-224106.json"
        ],
    },
    {
        "job_id": "1287578",
        "stage": "11-reference-partial",
        "checkpoints": [],
        "artifacts": ["outputs/eval/11_table4_reference4b.json"],
    },
    {
        "job_id": "1287579",
        "stage": "11-pruned-cancelled-pending",
        "checkpoints": [],
        "artifacts": [],
    },
    {
        "job_id": "1287580",
        "stage": "11-final-partial",
        "checkpoints": [],
        "artifacts": [
            "outputs/eval/11_table4_final.partial-preserved-20260606-224106.json"
        ],
    },
    {
        "job_id": "1290826_0",
        "stage": "12a",
        "checkpoints": [],
        "artifacts": ["outputs/eval/12_report_pre_kd.json"],
    },
    {
        "job_id": "1290826_1",
        "stage": "12a",
        "checkpoints": [],
        "artifacts": ["outputs/eval/12_report_kd15m.json"],
    },
    {
        "job_id": "1290826_2",
        "stage": "12a",
        "checkpoints": [],
        "artifacts": ["outputs/eval/12_report_final35m.json"],
    },
    {
        "job_id": "1290826_3",
        "stage": "12b",
        "checkpoints": [],
        "artifacts": ["outputs/eval/12_report_reference4b_native.json"],
    },
    {
        "job_id": "1290826_4",
        "stage": "12c",
        "checkpoints": [],
        "artifacts": ["outputs/eval/12_report_final35m_efficiency.json"],
    },
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda _value: None)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temp_name, path)
        path.chmod(0o644)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
        path.chmod(0o644)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _safetensors_parameter_count(path: Path) -> int:
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size))

    total = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        total += math.prod(int(dim) for dim in metadata["shape"])
    return total


def _throughput_by_batch(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["batch_size"]): row for row in rows}


def _matching_throughput(
    rows: list[dict[str, Any]],
    *,
    input_len: int = 4096,
    output_len: int = 256,
) -> dict[int, dict[str, Any]]:
    matched = [
        row
        for row in rows
        if int(row["input_len"]) == input_len
        and int(row["output_len"]) == output_len
    ]
    if not matched:
        raise ValueError(
            f"missing throughput measurements for {input_len}->{output_len}"
        )
    return _throughput_by_batch(matched)


def _candidate_rows() -> list[dict[str, Any]]:
    configs = _load_json(CANDIDATE_DIR / "candidates.json")
    evaluations = {
        row["candidate_id"]: row for row in _load_json(EVAL_DIR / "05_candidates.json")
    }
    selected = {
        row["candidate_id"]: rank
        for rank, row in enumerate(_load_json(EVAL_DIR / "06_top3.json"), start=1)
    }
    kd_round1 = {
        row["candidate_id"]: row for row in _load_json(EVAL_DIR / "08_kd_results.json")
    }
    parent = _load_json(EVAL_DIR / "01_baseline.json")
    parent_throughput = _matching_throughput(parent["throughput"])
    parent_b1 = parent_throughput[1]
    parent_b4 = parent_throughput[4]

    rows: list[dict[str, Any]] = []
    for config in configs:
        candidate_id = config["id"]
        evaluation = evaluations[candidate_id]
        throughput = _throughput_by_batch(evaluation["throughput"])
        b1 = throughput[1]
        b4 = throughput[4]
        checkpoint = _resolve(Path(evaluation["checkpoint"]))
        weight_path = checkpoint / "model.safetensors"
        kd = kd_round1.get(candidate_id, {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "selected_rank": selected.get(candidate_id),
                "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                "layers": config["layers"],
                "embedding_dim": config["embedding"],
                "ffn_dim": config["ffn"],
                "intended_mamba_heads": config["mamba_heads"],
                "intended_mamba_head_channels": config["mamba_head_channels"],
                "planned_param_count": config["param_count"],
                "effective_checkpoint_param_count": _safetensors_parameter_count(
                    weight_path
                ),
                "checkpoint_size_bytes": weight_path.stat().st_size,
                "pre_kd_lm_loss": evaluation["lm_loss"],
                "pre_kd_perplexity": evaluation["perplexity"],
                "kd15m_training_last_loss": kd.get("last_loss"),
                "tokens_per_second_b1": b1["tokens_per_second"],
                "tokens_per_second_b4": b4["tokens_per_second"],
                "relative_throughput_vs_parent_b4": (
                    b4["tokens_per_second"] / parent_b4["tokens_per_second"]
                ),
                "ttft_ms_b1": b1["time_to_first_token_ms"],
                "ttft_ms_b4": b4["time_to_first_token_ms"],
                "relative_ttft_vs_parent_b1": (
                    b1["time_to_first_token_ms"] / parent_b1["time_to_first_token_ms"]
                ),
                "peak_memory_gib_b1": b1["peak_memory_gib"],
                "peak_memory_gib_b4": b4["peak_memory_gib"],
                "mamba_structural_pruning_applied": False,
                "notes": (
                    "Saved checkpoint retains parent Mamba tensor dimensions; "
                    "planned Mamba widths and parameter count are search metadata."
                ),
            }
        )
    return rows


def _metric_payload(harness: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task, primary_key in TASKS:
        result = harness["results"][task]
        output[task] = {
            "primary_metric": primary_key.split(",", maxsplit=1)[0],
            "primary_value": result[primary_key],
            "primary_stderr": result.get(primary_key.replace(",none", "_stderr,none")),
            "acc": result.get("acc,none"),
            "acc_stderr": result.get("acc_stderr,none"),
            "acc_norm": result.get("acc_norm,none"),
            "acc_norm_stderr": result.get("acc_norm_stderr,none"),
            "sample_len": result.get("sample_len"),
        }
    return output


def _benchmark_models() -> list[dict[str, Any]]:
    stage09 = _load_json(EVAL_DIR / "09_final_eval.json")
    mini25 = _load_json(EVAL_DIR / "09_final_eval_mini_kd.json")[0]
    round30 = _load_json(EVAL_DIR / "09_final_eval_round2.json")[0]
    pre = _load_json(EVAL_DIR / "12_report_pre_kd.json")
    kd15 = _load_json(EVAL_DIR / "12_report_kd15m.json")
    final35 = _load_json(EVAL_DIR / "12_report_final35m.json")
    reference = _load_json(EVAL_DIR / "12_report_reference4b_native.json")

    specifications = [
        (
            "parent-8b",
            "Nemotron-H 8B parent",
            stage09[0]["checkpoint"],
            None,
            "outputs/eval/09_final_eval.json",
            "native",
            stage09[0]["harness"],
        ),
        (
            "cand-009-pre-kd",
            "cand-009 pre-KD",
            stage09[1]["checkpoint"],
            0,
            "outputs/eval/09_final_eval.json",
            "safe_torch",
            stage09[1]["harness"],
        ),
        (
            "cand-009-kd15m",
            "cand-009 KD 15M",
            stage09[2]["checkpoint"],
            15_000_576,
            "outputs/eval/09_final_eval.json",
            "safe_torch",
            stage09[2]["harness"],
        ),
        (
            "cand-016-pre-kd",
            "cand-016 pre-KD",
            pre["checkpoint"],
            0,
            "outputs/eval/12_report_pre_kd.json",
            pre["kernel_mode"],
            pre["harness"],
        ),
        (
            "cand-016-kd15m",
            "cand-016 KD 15M",
            kd15["checkpoint"],
            15_000_576,
            "outputs/eval/12_report_kd15m.json",
            kd15["kernel_mode"],
            kd15["harness"],
        ),
        (
            "cand-016-mini25m",
            "cand-016 mini-final ~25M",
            mini25["checkpoint"],
            25_000_960,
            "outputs/eval/09_final_eval_mini_kd.json",
            "safe_torch",
            mini25["harness"],
        ),
        (
            "cand-016-kd30m",
            "cand-016 round-2 ~30M",
            round30["checkpoint"],
            30_001_152,
            "outputs/eval/09_final_eval_round2.json",
            "safe_torch",
            round30["harness"],
        ),
        (
            "cand-016-final35m",
            "cand-016 mini-final2 ~35M",
            final35["checkpoint"],
            35_001_344,
            "outputs/eval/12_report_final35m.json",
            final35["kernel_mode"],
            final35["harness"],
        ),
        (
            "official-4b",
            "Official NVIDIA Nemotron-H 4B",
            reference["checkpoint"],
            None,
            "outputs/eval/12_report_reference4b_native.json",
            reference["kernel_mode"],
            reference["harness"],
        ),
    ]

    models: list[dict[str, Any]] = []
    for (
        model_id,
        label,
        checkpoint,
        lineage_tokens,
        source,
        kernel_mode,
        harness,
    ) in specifications:
        metrics = _metric_payload(harness)
        models.append(
            {
                "model_id": model_id,
                "label": label,
                "checkpoint": checkpoint,
                "lineage_kd_tokens": lineage_tokens,
                "source_artifact": source,
                "protocol": "lm-eval stable seven-task suite, zero-shot",
                "kernel_mode": kernel_mode,
                "mean_primary_accuracy": sum(
                    task["primary_value"] for task in metrics.values()
                )
                / len(metrics),
                "metrics": metrics,
            }
        )
    return models


def _benchmark_csv_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {
            key: model[key]
            for key in (
                "model_id",
                "label",
                "checkpoint",
                "lineage_kd_tokens",
                "source_artifact",
                "protocol",
                "kernel_mode",
                "mean_primary_accuracy",
            )
        }
        for task, _ in TASKS:
            metrics = model["metrics"][task]
            row[f"{task}_acc"] = metrics["acc"]
            row[f"{task}_acc_norm"] = metrics["acc_norm"]
            row[f"{task}_primary"] = metrics["primary_value"]
            row[f"{task}_stderr"] = metrics["primary_stderr"]
            row[f"{task}_samples"] = metrics["sample_len"]
        rows.append(row)
    return rows


def _recovery_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = (
        "cand-016-pre-kd",
        "cand-016-kd15m",
        "cand-016-mini25m",
        "cand-016-kd30m",
        "cand-016-final35m",
    )
    by_id = {model["model_id"]: model for model in models}
    pre = by_id[order[0]]
    rows: list[dict[str, Any]] = []
    for model_id in order:
        model = by_id[model_id]
        row: dict[str, Any] = {
            "model_id": model_id,
            "label": model["label"],
            "checkpoint": model["checkpoint"],
            "lineage_kd_tokens": model["lineage_kd_tokens"],
            "source_artifact": model["source_artifact"],
            "mean_primary_accuracy": model["mean_primary_accuracy"],
            "mean_delta_vs_pre_kd": (
                model["mean_primary_accuracy"] - pre["mean_primary_accuracy"]
            ),
        }
        for task, _ in TASKS:
            value = model["metrics"][task]["primary_value"]
            row[task] = value
            row[f"{task}_delta_vs_pre_kd"] = (
                value - pre["metrics"][task]["primary_value"]
            )
        rows.append(row)
    return rows


def _efficiency_rows() -> list[dict[str, Any]]:
    parent = _load_json(EVAL_DIR / "01_baseline.json")
    candidates = _load_json(EVAL_DIR / "05_candidates.json")
    selected_ids = {
        row["candidate_id"] for row in _load_json(EVAL_DIR / "06_top3.json")
    }
    final = _load_json(EVAL_DIR / "12_report_final35m_efficiency.json")

    rows: list[dict[str, Any]] = []

    def add_rows(
        model_id: str,
        checkpoint: str,
        kernel_mode: str,
        lm_loss: Any,
        perplexity: Any,
        throughput: list[dict[str, Any]],
        source: str,
    ) -> None:
        for measurement in throughput:
            rows.append(
                {
                    "model_id": model_id,
                    "checkpoint": checkpoint,
                    "kernel_mode": kernel_mode,
                    "lm_loss": lm_loss,
                    "perplexity": perplexity,
                    "input_len": measurement["input_len"],
                    "output_len": measurement["output_len"],
                    "batch_size": measurement["batch_size"],
                    "tokens_per_second": measurement["tokens_per_second"],
                    "time_to_first_token_ms": measurement[
                        "time_to_first_token_ms"
                    ],
                    "peak_memory_gib": measurement["peak_memory_gib"],
                    "source_artifact": source,
                }
            )

    add_rows(
        "parent-8b",
        parent["model"],
        "native",
        parent["lm_loss"]["loss"],
        parent["lm_loss"]["perplexity"],
        parent["throughput"],
        "outputs/eval/01_baseline.json",
    )
    for candidate in candidates:
        if candidate["candidate_id"] not in selected_ids:
            continue
        add_rows(
            f"{candidate['candidate_id']}-pre-kd",
            candidate["checkpoint"],
            "native",
            candidate["lm_loss"],
            candidate["perplexity"],
            candidate["throughput"],
            "outputs/eval/05_candidates.json",
        )
    add_rows(
        "cand-016-final35m",
        final["checkpoint"],
        final["kernel_mode"],
        final["lm_loss"]["loss"],
        final["lm_loss"]["perplexity"],
        final["throughput_results"],
        "outputs/eval/12_report_final35m_efficiency.json",
    )
    return rows


def _compact_stage09(output_path: Path) -> None:
    payload = _load_json(EVAL_DIR / "09_final_eval.json")
    removed = 0
    for model in payload:
        harness = model.get("harness", {})
        if "samples" in harness:
            harness.pop("samples")
            removed += 1
    if removed != len(payload):
        raise ValueError(
            f"expected samples in all {len(payload)} stage-09 entries; removed {removed}"
        )
    _write_json(output_path, payload)


def _slurm_jobs() -> list[dict[str, Any]]:
    command = [
        "sacct",
        "-j",
        ",".join(JOB_IDS),
        "--format=JobIDRaw,JobID,JobName%28,State,Elapsed,Timelimit,ExitCode,NodeList",
        "-X",
        "-n",
        "-P",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return [{"status": "unavailable", "error": str(exc), "command": command}]

    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) != 8:
            raise ValueError(f"unexpected sacct row: {line}")
        job_id_raw, job_id, job_name, state, elapsed, limit, exit_code, node = values
        rows.append(
            {
                "job_id_raw": job_id_raw,
                "job_id": job_id,
                "job_name": job_name,
                "state": state,
                "elapsed": elapsed,
                "time_limit": limit,
                "exit_code": exit_code,
                "node": node,
                "note": JOB_NOTES.get(job_id),
            }
        )
    return rows


def _checkpoint_manifest() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for checkpoint in sorted(path for path in CHECKPOINT_DIR.iterdir() if path.is_dir()):
        config_path = checkpoint / "config.json"
        weight_path = checkpoint / "model.safetensors"
        if not config_path.exists() or not weight_path.exists():
            continue
        config = _load_json(config_path)
        stat = weight_path.stat()
        records.append(
            {
                "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                "candidate_id": config.get("id"),
                "config_sha256": _sha256(config_path),
                "model_file": str(weight_path.relative_to(REPO_ROOT)),
                "model_size_bytes": stat.st_size,
                "model_mtime_epoch": stat.st_mtime,
                "effective_checkpoint_param_count": _safetensors_parameter_count(
                    weight_path
                ),
                "weights_sha256": None,
                "weights_sha256_note": (
                    "Not computed: checkpoint identity is recorded by exact path, "
                    "size, mtime, config hash, and tensor parameter count."
                ),
            }
        )
    return records


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "report",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    missing = [path for path in SOURCE_FILES if not (REPO_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required source artifacts: {missing}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "source_files": list(SOURCE_FILES),
                    "tasks": [task for task, _ in TASKS],
                },
                indent=2,
            )
        )
        return 0

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = _candidate_rows()
    candidate_fields = list(candidate_rows[0])
    _write_csv(output_dir / "candidate_table1.csv", candidate_rows, candidate_fields)
    _write_json(output_dir / "candidate_table1.json", candidate_rows)

    benchmark_models = _benchmark_models()
    benchmark_payload = {
        "schema_version": 1,
        "protocol": "lm-eval stable seven-task suite, zero-shot",
        "primary_metrics": {task: metric.split(",", maxsplit=1)[0] for task, metric in TASKS},
        "models": benchmark_models,
    }
    benchmark_rows = _benchmark_csv_rows(benchmark_models)
    _write_json(output_dir / "benchmark_compact.json", benchmark_payload)
    _write_csv(
        output_dir / "benchmark_compact.csv",
        benchmark_rows,
        list(benchmark_rows[0]),
    )

    recovery_rows = _recovery_rows(benchmark_models)
    _write_json(output_dir / "kd_recovery.json", recovery_rows)
    _write_csv(output_dir / "kd_recovery.csv", recovery_rows, list(recovery_rows[0]))

    efficiency_rows = _efficiency_rows()
    _write_json(output_dir / "efficiency.json", efficiency_rows)
    _write_csv(
        output_dir / "efficiency.csv",
        efficiency_rows,
        list(efficiency_rows[0]),
    )

    compact_stage09 = EVAL_DIR / "09_final_eval_compact.json"
    _compact_stage09(compact_stage09)

    generated_paths = [
        output_dir / "candidate_table1.csv",
        output_dir / "candidate_table1.json",
        output_dir / "benchmark_compact.csv",
        output_dir / "benchmark_compact.json",
        output_dir / "kd_recovery.csv",
        output_dir / "kd_recovery.json",
        output_dir / "efficiency.csv",
        output_dir / "efficiency.json",
        compact_stage09,
    ]
    manifest = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generation_seconds": time.time() - started,
        "repository": _git_state(),
        "source_artifacts": [_file_record(REPO_ROOT / path) for path in SOURCE_FILES],
        "generated_artifacts": [_file_record(path) for path in generated_paths],
        "slurm_jobs": _slurm_jobs(),
        "execution_lineage": list(EXECUTION_LINEAGE),
        "checkpoints": _checkpoint_manifest(),
        "limitations": [
            "Stage-04 Mamba structural pruning was not applied; effective checkpoints retain parent Mamba dimensions.",
            "Primary benchmark protocol is zero-shot, including MMLU.",
            "Stage-11 mixed-shot results are partial diagnostics and are not merged into benchmark_compact.",
            "Parent stage-01 LM loss is null; final35 and candidate LM-loss values come from their completed artifacts.",
        ],
    }
    _write_json(output_dir / "job_checkpoint_manifest.json", manifest)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "generated_artifacts": [
                    str(path.relative_to(REPO_ROOT))
                    for path in generated_paths
                    + [output_dir / "job_checkpoint_manifest.json"]
                ],
                "candidate_rows": len(candidate_rows),
                "benchmark_models": len(benchmark_models),
                "recovery_rows": len(recovery_rows),
                "efficiency_rows": len(efficiency_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
