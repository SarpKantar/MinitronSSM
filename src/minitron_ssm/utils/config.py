"""YAML config loading with typed dataclass schemas.

Every YAML file under ``configs/`` has a matching dataclass. Loaders
validate required keys, fill defaults, and reject unknown top-level
keys (a common source of silent misconfiguration).

Only the standard library + PyYAML are used so this module imports
cleanly on CPU-only dev machines.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional, Type, TypeVar, get_type_hints

import yaml

T = TypeVar("T")

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


# ---------------------------------------------------------------------------
# base.yaml
# ---------------------------------------------------------------------------


@dataclass
class Paths:
    outputs_dir: str
    scores_dir: str
    candidates_dir: str
    checkpoints_dir: str
    eval_dir: str
    hf_cache: Optional[str] = None


@dataclass
class Hardware:
    device: str
    num_gpus: int
    precision: str


@dataclass
class Models:
    parent: str
    reference_4b: str


@dataclass
class ParentArch:
    layers: int
    embedding: int
    ffn: int
    mamba_heads: int
    mamba_head_channels: int
    mamba_groups: int
    attention_layers: int
    vocab_size: int


@dataclass
class SeqLen:
    preferred: int
    fallback: int


@dataclass
class BaseConfig:
    run_name: str
    seed: int
    paths: Paths
    hardware: Hardware
    models: Models
    parent_arch: ParentArch
    seq_len: SeqLen


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------


@dataclass
class MixtureEntry:
    name: str
    hf_path: str
    weight: float
    hf_subset: Optional[str] = None


@dataclass
class ValidationData:
    hf_path: str
    num_tokens: int
    seed: int
    hf_subset: Optional[str] = None


@dataclass
class DataConfig:
    mixture: List[MixtureEntry]
    validation: ValidationData
    streaming: bool = True
    shuffle_buffer: int = 10_000


# ---------------------------------------------------------------------------
# importance.yaml
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    num_samples: int
    seq_len: int
    batch_size: int
    precision: str


@dataclass
class Aggregation:
    reduce_seq: str
    reduce_batch: str


@dataclass
class MambaScoring:
    score_source: str
    preserve_group_structure: bool
    head_channel_ranking: str


@dataclass
class ImportanceConfig:
    calibration: Calibration
    aggregation: Aggregation
    mamba: MambaScoring
    output_path: str


# ---------------------------------------------------------------------------
# search_space.yaml
# ---------------------------------------------------------------------------


@dataclass
class AnchorCandidate:
    id: str
    layers: int
    embedding: int
    ffn: int
    mamba_heads: int
    mamba_head_channels: int


@dataclass
class Grids:
    layers: List[int]
    embedding: List[int]
    ffn: List[int]
    mamba_heads: List[int]
    mamba_head_channels: List[int]


@dataclass
class DepthAblation:
    enabled: bool
    layers_options: List[int] = field(default_factory=list)


@dataclass
class SearchSpaceConfig:
    target_param_budget: int
    budget_tolerance: float
    max_candidates: int
    anchors: List[AnchorCandidate]
    grids: Grids
    depth_ablation: DepthAblation


# ---------------------------------------------------------------------------
# kd.yaml
# ---------------------------------------------------------------------------


@dataclass
class KDObjective:
    temperature: float
    alpha: float
    top_k: int


@dataclass
class KDBudget:
    top_candidates: int
    tokens_per_candidate: int
    smoke_test_tokens: int


@dataclass
class KDTraining:
    seq_len: int
    global_batch_size: int
    micro_batch_size: int
    grad_accumulation: Any
    activation_checkpointing: bool


@dataclass
class KDOptimizer:
    name: str
    lr: float
    weight_decay: float
    betas: List[float]


@dataclass
class KDScheduler:
    name: str
    warmup_steps: Optional[int]
    warmup_ratio: Optional[float]
    min_lr_ratio: float


@dataclass
class KDCache:
    path: str
    dtype: str


@dataclass
class KDConfig:
    objective: KDObjective
    budget: KDBudget
    training: KDTraining
    optimizer: KDOptimizer
    scheduler: KDScheduler
    mode: str
    cache: KDCache


# ---------------------------------------------------------------------------
# eval.yaml
# ---------------------------------------------------------------------------


@dataclass
class LMLossEval:
    seq_len: int
    num_tokens: int


@dataclass
class ThroughputEval:
    input_lens: List[int]
    output_lens: List[int]
    batch_sizes: List[int]
    warmup_iters: int
    measure_iters: int


@dataclass
class HarnessEval:
    tasks: List[str]
    optional_tasks: List[str]
    num_fewshot: int
    batch_size: Any


@dataclass
class EvalConfig:
    lm_loss: LMLossEval
    throughput: ThroughputEval
    lm_eval_harness: HarnessEval


# ---------------------------------------------------------------------------
# Loader plumbing
# ---------------------------------------------------------------------------


def _from_dict(cls: Type[T], data: Any) -> T:
    """Recursively build a dataclass tree from a nested dict.

    Raises a clear ``ValueError`` on missing required keys or unknown
    top-level keys.
    """
    if not is_dataclass(cls):
        return data  # primitive leaf

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected mapping for {cls.__name__}, got {type(data).__name__}"
        )

    hints = get_type_hints(cls)
    field_map = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(field_map)
    if unknown:
        raise ValueError(
            f"Unknown keys for {cls.__name__}: {sorted(unknown)}; "
            f"expected one of {sorted(field_map)}"
        )

    kwargs: dict[str, Any] = {}
    for name, f in field_map.items():
        if name in data:
            kwargs[name] = _convert(hints[name], data[name])
        elif f.default is not MISSING or f.default_factory is not MISSING:  # type: ignore[misc]
            continue  # use dataclass default
        else:
            raise ValueError(f"Missing required key {name!r} for {cls.__name__}")
    return cls(**kwargs)


def _convert(type_hint: Any, value: Any) -> Any:
    """Light-weight coercion: dataclass or List[dataclass] -> recurse; else passthrough."""
    origin = getattr(type_hint, "__origin__", None)
    args = getattr(type_hint, "__args__", ())

    if origin is list and args and is_dataclass(args[0]):
        return [_from_dict(args[0], v) for v in value]
    if is_dataclass(type_hint):
        return _from_dict(type_hint, value)
    return value


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def load_base(path: Optional[Path] = None) -> BaseConfig:
    return _from_dict(BaseConfig, _load_yaml(path or CONFIG_ROOT / "base.yaml"))


def load_data(path: Optional[Path] = None) -> DataConfig:
    return _from_dict(DataConfig, _load_yaml(path or CONFIG_ROOT / "data.yaml"))


def load_importance(path: Optional[Path] = None) -> ImportanceConfig:
    return _from_dict(
        ImportanceConfig, _load_yaml(path or CONFIG_ROOT / "importance.yaml")
    )


def load_search_space(path: Optional[Path] = None) -> SearchSpaceConfig:
    return _from_dict(
        SearchSpaceConfig, _load_yaml(path or CONFIG_ROOT / "search_space.yaml")
    )


def load_kd(path: Optional[Path] = None) -> KDConfig:
    return _from_dict(KDConfig, _load_yaml(path or CONFIG_ROOT / "kd.yaml"))


def load_eval(path: Optional[Path] = None) -> EvalConfig:
    return _from_dict(EvalConfig, _load_yaml(path or CONFIG_ROOT / "eval.yaml"))
