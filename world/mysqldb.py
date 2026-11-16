"""MySQL connection layer for BazaarDW (replaces DuckDB, see docs/decisions/0002-mysql.md).

Databases (per size ``small`` / ``full``):
- ``bazaar_<size>``: base tables (admin only).
- ``bazaar_<size>_analyst`` / ``_support`` / ``_finance``: views for each role.

The agent connects as a role user (``ql_analyst``, ``ql_support``, ``ql_finance``) that only has SELECT and
SHOW VIEW on its own role database, so base tables and other roles' columns are unreachable in the engine.
The admin account from ``.env`` is used only by the generator and the scoring of gold queries.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

import pymysql
import pymysql.cursors

ROLES = ("analyst", "support", "finance")
# same session settings for the agent and for gold queries: full precision for decimal division, UTC session time
SESSION_INIT = "SET SESSION div_precision_increment = 12, time_zone = '+00:00'"
ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_PATH)
    except ImportError:  # pragma: no cover
        pass


def settings() -> dict[str, Any]:
    _load_env()
    return {
        "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.environ.get("MYSQL_PORT", "3306")),
        "admin_user": os.environ.get("MYSQL_ADMIN_USER", "root"),
        "admin_password": os.environ.get("MYSQL_ADMIN_PASSWORD", ""),
    }


def db_name(size: str, role: str | None = None) -> str:
    if role in (None, "admin"):
        return f"bazaar_{size}"
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES} or 'admin'")
    return f"bazaar_{size}_{role}"


def role_user(role: str) -> str:
    return f"ql_{role}"


def role_password(role: str, create: bool = False) -> str:
    """Role passwords live in .env (MYSQL_<ROLE>_PASSWORD). With ``create``, a random one is generated and saved."""
    _load_env()
    key = f"MYSQL_{role.upper()}_PASSWORD"
    pw = os.environ.get(key, "")
    if pw or not create:
        return pw
    pw = secrets.token_urlsafe(18)
    text = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    lines = [ln for ln in text.splitlines() if not ln.startswith(f"{key}=")]
    lines.append(f"{key}={pw}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[key] = pw
    return pw


def admin_connect(database: str | None = None, **kw: Any) -> pymysql.connections.Connection:
    s = settings()
    return pymysql.connect(
        host=s["host"],
        port=s["port"],
        user=s["admin_user"],
        password=s["admin_password"],
        database=database,
        charset="utf8mb4",
        autocommit=True,
        init_command=SESSION_INIT,
        **kw,
    )


def role_connect(size: str, role: str, read_timeout: int | None = None) -> pymysql.connections.Connection:
    """Read-only connection as the role user, with the role database as the default schema."""
    if role == "admin":
        return admin_connect(db_name(size))
    s = settings()
    con = pymysql.connect(
        host=s["host"],
        port=s["port"],
        user=role_user(role),
        password=role_password(role),
        database=db_name(size, role),
        charset="utf8mb4",
        autocommit=True,
        read_timeout=read_timeout,
        init_command=SESSION_INIT,
    )
    with con.cursor() as cur:
        cur.execute("SET SESSION TRANSACTION READ ONLY")
    return con


def fetchall(con: pymysql.connections.Connection, sql: str, args: Any = None) -> list[tuple[Any, ...]]:
    with con.cursor() as cur:
        cur.execute(sql, args)
        return list(cur.fetchall())


def fetch_with_columns(
    con: pymysql.connections.Connection,
    sql: str,
    limit: int | None = None,
) -> tuple[list[str], list[tuple[Any, ...]], int]:
    """(columns, rows, total) streaming the result; at most ``limit`` rows kept, but all rows counted."""
    cur = con.cursor(pymysql.cursors.SSCursor)
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows: list[tuple[Any, ...]] = []
        total = 0
        for r in cur:
            total += 1
            if limit is None or len(rows) < limit:
                rows.append(r)
        return cols, rows, total
    finally:
        cur.close()


def available() -> bool:
    try:
        admin_connect().close()
        return True
    except pymysql.MySQLError:
        return False
