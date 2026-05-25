"""Idempotent logger setup."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import STATE_DIR

_CONFIGURED: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger
    level_name = os.environ.get("APEX_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    try:
        log_path = Path(STATE_DIR()) / "apex.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass

    _CONFIGURED.add(name)
    return logger
