"""SQL guard and output PII scan (CLAUDE.md 7.2), MySQL 8 dialect.

The guard is the code-level permission layer: every query the model writes passes :func:`check_sql`
before it reaches MySQL. Rejections carry a ``rule`` id (for traces and F7 counting) and a message the
model can act on. The connection layer adds a second, physical layer: the agent connects as a role user with
SELECT-only grants on its own role database, in a READ ONLY session, with ``max_execution_time`` set.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pymysql
import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError, SqlglotError
from sqlglot.expressions.core import Expression
from sqlglot.optimizer.qualify import qualify

from .config import SqlRuntime

DIALECT = "mysql"

BANNED_STATEMENTS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Pragma,
    exp.Set,
    exp.Use,
    exp.Describe,
    exp.TruncateTable,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Grant,
    exp.Kill,
    exp.Show,
    exp.LoadData,
)
# functions that read files, stall the server, take locks or reveal server/session state
FORBIDDEN_FUNCTIONS = {
    "load_file",
    "sleep",
    "benchmark",
    "get_lock",
    "release_lock",
    "release_all_locks",
    "is_free_lock",
    "is_used_lock",
    "master_pos_wait",
    "source_pos_wait",
    "wait_for_executed_gtid_set",
    "sys_exec",
    "sys_eval",
    "user",
    "current_user",
    "session_user",
    "system_user",
    "database",
    "schema",
    "version",
    "connection_id",
    "found_rows",
    "row_count",
    "last_insert_id",
    "load_extension",
    "get_format_dummy",
    "statement_digest",
    "ps_current_thread_id",
    "ps_thread_id",
    "format_bytes",
    "format_pico_time",
}
SYSTEM_SCHEMAS = {"information_schema", "mysql", "performance_schema", "sys"}
URL_RE = re.compile(r"^\s*(https?|s3|gcs|gs|az|azure|file|ftp)://", re.I)
PATH_RE = re.compile(r"[\\/]|\.(csv|parquet|json|txt|db|sql|sqlite|xlsx|ini|cnf|log)$", re.I)


@dataclass
class GuardResult:
    ok: bool
    rule: str = "ok"
    message: str = ""
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    normalized_sql: str = ""


def _reject(rule: str, message: str) -> GuardResult:
    return GuardResult(ok=False, rule=rule, message=message)


@dataclass(frozen=True)
class RoleSchema:
    role: str
    tables: dict[str, dict[str, str]]  # table -> {column: type}
    base_tables: dict[str, list[str]]  # full warehouse schema, to explain forbidden objects

    def table_names(self) -> list[str]:
        return sorted(self.tables)


@lru_cache(maxsize=16)
def role_schema(size: str, role: str) -> RoleSchema:
    """Tables and columns of the role database (it contains only what the role may see)."""
    from bazaar_world.mysqldb import admin_connect, db_name, fetchall
    from bazaar_world.schema import TABLES

    con = admin_connect()
    try:
        rows = fetchall(
            con,
            "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = %s ORDER BY TABLE_NAME, ORDINAL_POSITION",
            (db_name(size, role),),
        )
    finally:
        con.close()
    tables: dict[str, dict[str, str]] = {}
    for t, c, d in rows:
        tables.setdefault(t.lower(), {})[c.lower()] = d
    base = {t.name: t.column_names for t in TABLES}
    return RoleSchema(role=role, tables=tables, base_tables=base)


ROLE_WITH_ACCESS = {
    ("customers", "full_name"): "support",
    ("support_messages", "*"): "support",
}


def _who_can(table: str, column: str | None) -> str:
    if column and (table, column) in ROLE_WITH_ACCESS:
        return ROLE_WITH_ACCESS[(table, column)]
    if (table, "*") in ROLE_WITH_ACCESS:
        return ROLE_WITH_ACCESS[(table, "*")]
    return "none (PII is not exposed to any agent role)"


def _closest(name: str, options: list[str], n: int = 3) -> list[str]:
    return difflib.get_close_matches(name, options, n=n, cutoff=0.4)


def _is_aggregate_query(tree: Expression) -> bool:
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if sel is None:
        return False
    if sel.args.get("group"):
        return True
    return any(isinstance(e.unalias(), exp.AggFunc) or e.find(exp.AggFunc) for e in sel.expressions)


def _has_limit(tree: Expression) -> bool:
    return tree.args.get("limit") is not None


def check_sql(sql: str, schema: RoleSchema) -> GuardResult:
    """Static checks; does not touch the database."""
    if not sql or not sql.strip():
        return _reject("empty", "The query is empty. Send one SELECT statement.")
    if re.search(r"\binto\s+(outfile|dumpfile)\b", sql, re.I):
        return _reject(
            "file_access", "Writing query results to files (INTO OUTFILE/DUMPFILE) is not allowed."
        )
    try:
        stmts = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except (ParseError, SqlglotError) as e:
        return _reject(
            "parse_error", f"The SQL does not parse (MySQL dialect): {str(e).splitlines()[0][:200]}"
        )
    if len(stmts) != 1:
        return _reject("multiple_statements", f"Send exactly one statement; got {len(stmts)}.")
    tree = stmts[0]
    if isinstance(tree, BANNED_STATEMENTS) or not isinstance(tree, (exp.Select, exp.SetOperation)):
        kind = type(tree).__name__.upper()
        return _reject(
            "non_select",
            f"Only read-only SELECT queries are allowed (got {kind}). "
            "The database is read-only; writes and settings are never permitted.",
        )
    for node in tree.walk():
        if isinstance(node, BANNED_STATEMENTS):
            return _reject(
                "non_select",
                f"Statements of type {type(node).__name__.upper()} are not allowed inside a query.",
            )
        if isinstance(node, exp.Into):
            return _reject("non_select", "SELECT ... INTO is not allowed.")
        if isinstance(node, (exp.SessionParameter, exp.Parameter)) or (
            isinstance(node, exp.Var) and str(node.this).startswith("@")
        ):
            return _reject("system_catalog", "Session and server variables (@x, @@x) are not available.")
    for node in tree.find_all(exp.Func):
        fname = (node.sql_name() if not isinstance(node, exp.Anonymous) else str(node.this)).lower()
        rendered = node.sql(dialect=DIALECT).split("(", 1)[0].strip().lower()  # typed nodes, e.g. VERSION()
        if (
            fname in FORBIDDEN_FUNCTIONS
            or rendered in FORBIDDEN_FUNCTIONS
            or isinstance(node, (exp.CurrentUser, exp.CurrentSchema))
        ):
            return _reject(
                "file_access",
                f"Function {fname}() reads files, stalls the server or reveals server "
                "state, and is not allowed. Query the warehouse tables instead.",
            )
    for lit in tree.find_all(exp.Literal):
        if lit.is_string and URL_RE.match(str(lit.this)):
            return _reject("file_access", "URLs are not allowed in queries.")
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables: list[str] = []
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            return _reject("file_access", "Table functions are not allowed; use table names.")
        name = t.name
        if PATH_RE.search(name):
            return _reject(
                "file_access", f"'{name}' looks like a file path; only warehouse tables are allowed."
            )
        db = (t.args.get("db").name if t.args.get("db") else "").lower()
        catalog = (t.args.get("catalog").name if t.args.get("catalog") else "").lower()
        if db or catalog:
            if db in SYSTEM_SCHEMAS or catalog in SYSTEM_SCHEMAS:
                return _reject(
                    "system_catalog",
                    "System databases are not available. Use search_schema or "
                    "describe_table to explore the schema.",
                )
            return _reject(
                "schema_reference",
                f"Do not qualify table names with a database ('{db or catalog}."
                f"{name}'). Your role database is already active: use {name}.",
            )
        lname = name.lower()
        if lname in cte_names:
            continue
        if lname in SYSTEM_SCHEMAS:
            return _reject("system_catalog", "System tables are not available. Use describe_table.")
        if lname not in schema.tables:
            if lname in schema.base_tables:
                return _reject(
                    "forbidden_table",
                    f"Table {lname} is not available to role {schema.role}. "
                    f"Role with access: {_who_can(lname, None)}.",
                )
            close = _closest(lname, schema.table_names())
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return _reject(
                "unknown_table", f"Table {name} does not exist.{hint} Use search_schema to find tables."
            )
        tables.append(lname)
    mapping = {t: dict(cols) for t, cols in schema.tables.items()}
    columns: list[str] = []
    try:
        q = qualify(
            tree.copy(),
            schema=mapping,
            dialect=DIALECT,
            validate_qualify_columns=True,
            quote_identifiers=False,
            identify=False,
        )
        alias_to_table: dict[str, str] = {}
        for t in q.find_all(exp.Table):
            if t.name.lower() in schema.tables:
                alias_to_table[(t.alias_or_name or t.name).lower()] = t.name.lower()
        for c in q.find_all(exp.Column):
            tbl = alias_to_table.get((c.table or "").lower())
            if tbl:
                columns.append(f"{tbl}.{c.name.lower()}")
        # defence in depth: sqlglot can mis-resolve scopes (e.g. LATERAL); re-check raw column names
        extra = _fallback_columns(tree, tables, schema)
        if isinstance(extra, GuardResult):
            return extra
    except OptimizeError as e:
        columns = []
        msg = str(e)
        m = re.search(r"[Cc]olumn '?\"?`?([A-Za-z_][\w]*)", msg)
        col = m.group(1).lower() if m else ""
        if col:
            owners = [t for t in tables if col in schema.base_tables.get(t, [])]
            if owners and all(col not in schema.tables.get(t, {}) for t in owners):
                return _reject(
                    "forbidden_column",
                    f"Column {owners[0]}.{col} is not available to role "
                    f"{schema.role}. Role with access: {_who_can(owners[0], col)}.",
                )
            known = sorted({c for t in tables for c in schema.tables.get(t, {})})
            close = _closest(col, known)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return _reject(
                "unknown_column",
                f"Column {col} does not exist in the referenced tables.{hint} "
                "Use describe_table to see the columns.",
            )
        fb = _fallback_columns(tree, tables, schema)
        if isinstance(fb, GuardResult):
            return fb
    except SqlglotError:
        fb = _fallback_columns(tree, tables, schema)
        if isinstance(fb, GuardResult):
            return fb
    return GuardResult(
        ok=True,
        tables=sorted(set(tables)),
        columns=sorted(set(columns)),
        normalized_sql=tree.sql(dialect=DIALECT),
    )


def _fallback_columns(tree: Expression, tables: list[str], schema: RoleSchema) -> list[str] | GuardResult:
    known = {c for t in tables for c in schema.tables.get(t, {})}
    aliases = {a.alias.lower() for a in tree.find_all(exp.Alias)} | {
        c.alias_or_name.lower() for c in tree.find_all(exp.CTE)
    }
    for c in tree.find_all(exp.Column):
        n = c.name.lower()
        if n in known or n in aliases or not n:
            continue
        for t in tables:
            if n in schema.base_tables.get(t, []):
                return _reject(
                    "forbidden_column",
                    f"Column {t}.{n} is not available to role {schema.role}. "
                    f"Role with access: {_who_can(t, n)}.",
                )
    return []


# --------------------------------------------------------------------------- connection-level guard
def open_explain_connection(size: str, role: str) -> pymysql.connections.Connection:
    """Connection used only for EXPLAIN of SQL the guard has already approved.

    MySQL refuses EXPLAIN on a view unless the user may also read the view's base tables, which the role users
    deliberately cannot. EXPLAIN does not execute the query; this connection is the admin account pointed at the
    role database, in a READ ONLY session, and it never receives SQL that failed :func:`check_sql`.
    """
    from bazaar_world.mysqldb import admin_connect, db_name

    con = admin_connect(db_name(size, role))
    with con.cursor() as cur:
        cur.execute("SET SESSION TRANSACTION READ ONLY")
    return con


def open_role_connection(size: str, role: str, rt: SqlRuntime) -> pymysql.connections.Connection:
    """Role user (SELECT-only grants), READ ONLY session, statement time limit (rule 1)."""
    from bazaar_world.mysqldb import role_connect

    con = role_connect(size, role, read_timeout=int(rt.timeout_s * 3) + 5)
    with con.cursor() as cur:
        cur.execute(f"SET SESSION max_execution_time = {int(rt.timeout_s * 1000)}")
    return con


class QueryTimeout(RuntimeError):
    pass


TIMEOUT_CODES = {3024, 1317, 2013}  # max_execution_time exceeded, query interrupted, lost connection


def execute_with_timeout(
    con: pymysql.connections.Connection,
    sql: str,
    timeout_s: float,
    max_rows: int,
) -> tuple[list[str], list[tuple[Any, ...]], int]:
    """Run ``sql`` (stopped by MySQL after ``timeout_s``). Returns (columns, first ``max_rows`` rows, total rows)."""
    from bazaar_world.mysqldb import fetch_with_columns

    try:
        return fetch_with_columns(con, sql, limit=max_rows)
    except pymysql.MySQLError as e:
        if e.args and e.args[0] in TIMEOUT_CODES:
            raise QueryTimeout(f"query exceeded {timeout_s:.0f} s and was stopped") from e
        raise


def explain_cardinality(con: pymysql.connections.Connection, sql: str) -> int:
    """Estimated result rows from EXPLAIN: product of rows x filtered over the outer query's tables."""
    with con.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(f"EXPLAIN {sql}")
        plan = cur.fetchall()
    est = 1.0
    outer = [r for r in plan if r.get("id") == 1 and r.get("select_type") in ("SIMPLE", "PRIMARY")]
    for r in outer or plan[:1]:
        rows = float(r.get("rows") or 1)
        filtered = float(r.get("filtered") or 100.0)
        est *= max(1.0, rows * filtered / 100.0)
    return int(min(est, 10**12))


def cost_check(con: pymysql.connections.Connection, sql: str, rt: SqlRuntime) -> GuardResult | None:
    """Reject large unbounded result sets (no LIMIT and no aggregation above the row threshold)."""
    tree = sqlglot.parse_one(sql, read=DIALECT)
    if _has_limit(tree) or _is_aggregate_query(tree):
        return None
    est = explain_cardinality(con, sql)
    if est > rt.explain_row_threshold:
        return _reject(
            "too_many_rows",
            f"The query would return about {est:,} rows (limit "
            f"{rt.explain_row_threshold:,} without LIMIT or aggregation). "
            "Aggregate the data or add a LIMIT.",
        )
    return None


# --------------------------------------------------------------------------- output PII scan
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\d.])(?:\+?880[\s-]?1[3-9]\d{2}[\s-]?\d{6}|01[3-9]\d{2}[\s-]?\d{6})(?![\d.])")


def scan_output(text: str) -> tuple[str, list[str]]:
    """Redact emails and Bangladeshi phone numbers. Returns (redacted text, kinds found)."""
    found: list[str] = []
    if EMAIL_RE.search(text):
        found.append("email")
        text = EMAIL_RE.sub("[REDACTED EMAIL]", text)
    if PHONE_RE.search(text):
        found.append("phone")
        text = PHONE_RE.sub("[REDACTED PHONE]", text)
    return text, found
