"""Logging utilities."""
from __future__ import annotations

import logging
from typing import Optional


_LOGGER_CACHE: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    _LOGGER_CACHE[name] = logger
    return logger


__all__ = ["get_logger"]
