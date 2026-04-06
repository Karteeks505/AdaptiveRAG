from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["paths"]["data_dir"])
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def resolve_data_path(cfg: dict[str, Any], relative: str) -> Path:
    return (data_root(cfg) / relative).resolve()
