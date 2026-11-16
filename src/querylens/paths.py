"""Repository paths (all relative to the repository root)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("QUERYLENS_DATA", ROOT / "data"))
WORLD = ROOT / "world"
SEMANTIC = ROOT / "semantic"
CONFIGS = ROOT / "configs"
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"
RUBRICS = ROOT / "rubrics"
PROMPTS = Path(__file__).resolve().parent / "prompts"


def db_name(size: str, role: str = "analyst") -> str:
    """MySQL database name for a size and role (``bazaar_full_analyst``; ``admin`` -> base ``bazaar_full``)."""
    from bazaar_world.mysqldb import db_name as _db_name

    return _db_name(size, None if role == "admin" else role)
