"""Shared fixtures. Tests use the small warehouse; it is built once if missing."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _ensure_small() -> Path:
    path = DATA / "bazaar_small.duckdb"
    role = DATA / "bazaar_small__analyst.duckdb"
    if not path.exists() or not role.exists():
        from bazaar_world.generate import build

        build("small", 42, DATA, write_metadata=False)
    return path


@pytest.fixture(scope="session")
def small_db() -> Path:
    return _ensure_small()


@pytest.fixture(scope="session")
def small_con(small_db: Path):
    con = duckdb.connect(str(small_db), read_only=True)
    yield con
    con.close()


@pytest.fixture(scope="session")
def analyst_db(small_db: Path) -> Path:
    return small_db.parent / "bazaar_small__analyst.duckdb"


@pytest.fixture(scope="session")
def full_db() -> Path:
    path = DATA / "bazaar_full.duckdb"
    if not path.exists():
        pytest.skip("full-size warehouse not built (make world SIZE=full)")
    return path
