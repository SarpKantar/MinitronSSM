"""Shared CLI helpers for scripts/01..10.

Keeps ``argparse`` boilerplate out of every script.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "configs" / "base.yaml",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configs and print the planned actions, then exit.",
    )
    return p


def _to_dict(obj):
    if is_dataclass(obj):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def print_resolved(label: str, cfg) -> None:
    print(f"=== resolved {label} ===")
    print(json.dumps(_to_dict(cfg), indent=2, default=str))
