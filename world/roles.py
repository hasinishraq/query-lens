"""Role databases and users in MySQL: physical enforcement of the role views (CLAUDE.md 6.7, rules 1-2).

For each size, ``bazaar_<size>_<role>`` holds only views over the base tables. The role user ``ql_<role>`` gets
SELECT and SHOW VIEW on that database and nothing else: no base tables, no FILE privilege (so no
INTO OUTFILE / LOAD_FILE / LOAD DATA), no writes. The guard enforces the same rules again in code.
"""

from __future__ import annotations

from pathlib import Path

import pymysql

from .mysqldb import ROLES, admin_connect, db_name, fetchall, role_password, role_user

VIEWS_SQL = Path(__file__).resolve().parent / "views.sql"
HOSTS = ("localhost", "127.0.0.1")


def role_columns(con: pymysql.connections.Connection, database: str) -> dict[str, list[tuple[str, str]]]:
    rows = fetchall(
        con,
        "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = %s ORDER BY TABLE_NAME, ORDINAL_POSITION",
        (database,),
    )
    out: dict[str, list[tuple[str, str]]] = {}
    for t, c, d in rows:
        out.setdefault(t, []).append((c, d))
    return out


def _statements(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def ensure_users(con: pymysql.connections.Connection) -> None:
    with con.cursor() as cur:
        for role in ROLES:
            pw = role_password(role, create=True)
            for host in HOSTS:
                cur.execute(f"CREATE USER IF NOT EXISTS '{role_user(role)}'@'{host}' IDENTIFIED BY %s", (pw,))
                cur.execute(f"ALTER USER '{role_user(role)}'@'{host}' IDENTIFIED BY %s", (pw,))


def build_role_dbs(size: str) -> dict[str, str]:
    """(Re)create the three role databases of a size with their views, users and grants."""
    base = db_name(size)
    names = {r: db_name(size, r) for r in ROLES}
    con = admin_connect()
    try:
        ensure_users(con)
        with con.cursor() as cur:
            for d in names.values():
                cur.execute(f"DROP DATABASE IF EXISTS `{d}`")
                cur.execute(f"CREATE DATABASE `{d}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_as_cs")
            sql = VIEWS_SQL.read_text(encoding="utf-8").replace("{base}", f"`{base}`")
            for r, d in names.items():
                sql = sql.replace("{" + r + "}", f"`{d}`")
            for stmt in _statements(sql):
                cur.execute(stmt)
            for r, d in names.items():
                for host in HOSTS:
                    cur.execute(f"GRANT SELECT, SHOW VIEW ON `{d}`.* TO '{role_user(r)}'@'{host}'")
            cur.execute("FLUSH PRIVILEGES")
    finally:
        con.close()
    return names
