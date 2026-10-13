"""Centralized logging setup for SIGN-X."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@lru_cache(maxsize=None)
def get_logger(name: str = "signx") -> logging.Logger:
    """Return a configured logger, memoized per name so handlers are never
    attached twice (which would duplicate every log line)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        from app.config import settings

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        logger.propagate = False
    return logger
