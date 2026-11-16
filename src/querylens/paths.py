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


def db_path(size: str, role: str = "analyst") -> Path:
    if role == "admin":
        return DATA / f"bazaar_{size}.duckdb"
    return DATA / f"bazaar_{size}__{role}.duckdb"
