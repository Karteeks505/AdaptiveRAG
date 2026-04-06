from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def write_manifest(
    run_dir: Path,
    config_path: Path,
    config_dict: dict[str, Any],
    data_files: list[Path],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = _sha256_text(json.dumps(config_dict, sort_keys=True))
    data_hashes = {str(p): _sha256_file(p) for p in data_files if p.exists()}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_sha256": cfg_hash,
        "data_sha256": data_hashes,
    }
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
