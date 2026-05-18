"""Minimal stdlib-only logging helper.

We avoid pulling in heavier loggers (rich, loguru, wandb) at the
skeleton stage so the package imports cleanly on a fresh CPU venv.
Stage scripts can swap this for wandb/tensorboard later.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str = "minitron_ssm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_level(level: int, name: Optional[str] = None) -> None:
    logging.getLogger(name if name else "minitron_ssm").setLevel(level)
