"""Configuration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with open(Path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class ExperimentConfig:
    """Container for experiment level settings."""

    tracker: Dict[str, Any]
    data: Dict[str, Any]
    training: Dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        cfg = load_config(path)
        return cls(
            tracker=cfg.get("tracker", {}),
            data=cfg.get("data", {}),
            training=cfg.get("training", {}),
        )


__all__ = ["load_config", "ExperimentConfig"]
