"""Role databases: physical enforcement of the role views.

DuckDB falls back to the ``main`` schema for unqualified names even when ``search_path`` points to a
role schema, so views alone cannot stop a query from reaching a base table. Each role therefore gets
its own DuckDB file whose ``main`` schema holds *only* the role's view contents
(``data/bazaar_<size>__<role>.duckdb``). The agent opens that file read-only; base tables and the
other roles' columns do not exist there. The guard enforces the same rules again in code.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

ROLES: dict[str, str] = {"analyst": "v_analyst", "support": "v_support", "finance": "v_finance"}


def role_db_path(data_dir: Path, size: str, role: str) -> Path:
    if role == "admin":
        return data_dir / f"bazaar_{size}.duckdb"
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(ROLES)} or 'admin'")
    return data_dir / f"bazaar_{size}__{role}.duckdb"


def role_tables(con: duckdb.DuckDBPyConnection, schema: str) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = ? ORDER BY table_name",
        [schema],
    ).fetchall()
    return [r[0] for r in rows]


def role_columns(con: duckdb.DuckDBPyConnection, schema: str) -> dict[str, list[tuple[str, str]]]:
    rows = con.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = ? "
        "ORDER BY table_name, ordinal_position",
        [schema],
    ).fetchall()
    out: dict[str, list[tuple[str, str]]] = {}
    for t, c, d in rows:
        out.setdefault(t, []).append((c, d))
    return out


def build_role_dbs(base_path: Path, size: str) -> dict[str, Path]:
    data_dir = base_path.parent
    out: dict[str, Path] = {}
    for role, schema in ROLES.items():
        path = role_db_path(data_dir, size, role)
        if path.exists():
            path.unlink()
        con = duckdb.connect(str(path))
        try:
            con.execute(f"ATTACH '{base_path.as_posix()}' AS base (READ_ONLY)")
            tables = [
                r[0]
                for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'base' "
                    "AND table_schema = ? ORDER BY table_name",
                    [schema],
                ).fetchall()
            ]
            for t in tables:
                con.execute(f'CREATE TABLE main."{t}" AS SELECT * FROM base.{schema}."{t}"')
            con.execute("DETACH base")
            con.execute("CHECKPOINT")
        finally:
            con.close()
        out[role] = path
    return out
