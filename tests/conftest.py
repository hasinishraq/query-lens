"""Shared fixtures. Tests use the small warehouse in MySQL (``bazaar_small``); it is built once if missing.

MySQL settings come from ``.env`` (MYSQL_HOST, MYSQL_PORT, MYSQL_ADMIN_USER, MYSQL_ADMIN_PASSWORD). Tests that need
the database are skipped when MySQL is not reachable.
"""

from __future__ import annotations

import pytest

from bazaar_world.mysqldb import admin_connect, available, db_name, fetchall


def _exists(size: str) -> bool:
    con = admin_connect()
    try:
        names = (db_name(size), db_name(size, "analyst"), db_name(size, "support"), db_name(size, "finance"))
        rows = fetchall(
            con,
            "SELECT COUNT(*) FROM information_schema.SCHEMATA WHERE SCHEMA_NAME IN (%s, %s, %s, %s)",
            names,
        )
    finally:
        con.close()
    return rows[0][0] == 4


@pytest.fixture(scope="session")
def mysql_up() -> bool:
    if not available():
        pytest.skip("MySQL is not reachable (check MYSQL_* in .env)")
    return True


@pytest.fixture(scope="session")
def small_db(mysql_up: bool) -> str:
    """Name of the small base database; built if missing."""
    if not _exists("small"):
        from pathlib import Path

        from bazaar_world.generate import build

        build("small", 42, Path("data"), write_metadata=False)
    return db_name("small")


@pytest.fixture(scope="session")
def small_con(small_db: str):
    con = admin_connect(small_db)
    yield con
    con.close()


@pytest.fixture(scope="session")
def analyst_db(small_db: str) -> str:
    """The analyst role database of the small size."""
    return db_name("small", "analyst")


@pytest.fixture(scope="session")
def full_db(mysql_up: bool) -> str:
    if not _exists("full"):
        pytest.skip("full-size warehouse not built (python -m bazaar_world.generate --size full)")
    return db_name("full")
