"""SQL guard and output PII scan (CLAUDE.md 7.2).

The guard is the code-level permission layer: every query the model writes passes :func:`check_sql`
before it reaches DuckDB. Rejections carry a ``rule`` id (for traces and F7 counting) and a message the
model can act on. Connection-level limits (read-only, no external access, memory, timeout) are applied
in :func:`open_role_connection` and :func:`execute_with_timeout`.
"""

from __future__ import annotations

import difflib
import json
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError, SqlglotError
from sqlglot.expressions.core import Expression
from sqlglot.optimizer.qualify import qualify

from .config import SqlRuntime

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
    exp.Attach,
    exp.Detach,
    exp.Install,
    exp.Use,
    exp.Describe,
    exp.Summarize,
    exp.TruncateTable,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Grant,
    exp.Kill,
)
FILE_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "parquet_scan",
    "read_json",
    "read_json_auto",
    "read_json_objects",
    "read_ndjson",
    "read_ndjson_auto",
    "read_text",
    "read_blob",
    "glob",
    "csv_scan",
    "sniff_csv",
    "parquet_metadata",
    "parquet_schema",
    "iceberg_scan",
    "delta_scan",
    "sqlite_scan",
    "postgres_scan",
    "mysql_scan",
    "read_xlsx",
    "query_table",
    "query",
    "json_execute_serialized_sql",
    "getenv",
    "duckdb_secrets",
    "duckdb_settings",
    "duckdb_extensions",
    "duckdb_databases",
    "pragma_version",
    "load_extension",
    "install_extension",
    "current_setting",
}
SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "system", "temp"}
URL_RE = re.compile(r"^\s*(https?|s3|gcs|gs|az|azure|hf|r2|file)://", re.I)
PATH_RE = re.compile(r"\.(csv|parquet|json|ndjson|jsonl|txt|db|duckdb|sqlite|xlsx|gz|zst)\b", re.I)


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
def role_schema(db_path: str, role: str) -> RoleSchema:
    """Read the role database's tables and columns (role DBs contain only what the role may see)."""
    from bazaar_world.schema import TABLES

    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='main' ORDER BY table_name, ordinal_position"
        ).fetchall()
    finally:
        con.close()
    tables: dict[str, dict[str, str]] = {}
    for t, c, d in rows:
        tables.setdefault(t, {})[c] = d
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
    try:
        stmts = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except (ParseError, SqlglotError) as e:
        return _reject(
            "parse_error", f"The SQL does not parse (DuckDB dialect): {str(e).splitlines()[0][:200]}"
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
    # file / URL readers and system functions
    for node in tree.find_all(exp.Func):
        fname = (node.sql_name() if not isinstance(node, exp.Anonymous) else str(node.this)).lower()
        if (
            fname in FILE_FUNCTIONS
            or isinstance(node, (exp.ReadCSV,))
            or fname.startswith(("read_", "duckdb_"))
        ):
            return _reject(
                "file_access",
                f"Function {fname}() reads files, URLs or system state and is not "
                "allowed. Query the warehouse tables instead.",
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
        if t.this.quoted and (PATH_RE.search(name) or "/" in name or "\\" in name):
            return _reject(
                "file_access", f"'{name}' looks like a file path; only warehouse tables are allowed."
            )
        db = (t.args.get("db").name if t.args.get("db") else "").lower()
        catalog = (t.args.get("catalog").name if t.args.get("catalog") else "").lower()
        if db or catalog:
            if db in SYSTEM_SCHEMAS or catalog in SYSTEM_SCHEMAS:
                return _reject(
                    "system_catalog",
                    "System catalogs are not available. Use search_schema or "
                    "describe_table to explore the schema.",
                )
            return _reject(
                "schema_reference",
                f"Do not qualify table names with a schema ('{db or catalog}."
                f"{name}'). Your role schema is already active: use {name}.",
            )
        lname = name.lower()
        if lname in cte_names:
            continue
        if lname.startswith(("duckdb_", "pragma_", "sqlite_")):
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
    # columns: resolve aliases and * with the role schema
    mapping = {t: dict(cols) for t, cols in schema.tables.items()}
    columns: list[str] = []
    try:
        q = qualify(
            tree.copy(),
            schema=mapping,
            dialect="duckdb",
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
        m = re.search(r"[Cc]olumn '?\"?([A-Za-z_][\w]*)", msg)
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
        # unresolvable for other reasons (complex DuckDB syntax): fall back to a conservative name check
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
        normalized_sql=tree.sql(dialect="duckdb"),
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
def open_role_connection(db_path: str | Path, rt: SqlRuntime) -> duckdb.DuckDBPyConnection:
    """Read-only connection with external access disabled and a memory cap (rule 1)."""
    return duckdb.connect(
        str(db_path),
        read_only=True,
        config={
            "enable_external_access": False,
            "memory_limit": rt.memory_limit,
            "threads": rt.threads,
            "autoload_known_extensions": False,
            "autoinstall_known_extensions": False,
            "lock_configuration": True,
        },
    )


class QueryTimeout(RuntimeError):
    pass


def execute_with_timeout(
    con: duckdb.DuckDBPyConnection, sql: str, timeout_s: float, max_rows: int
) -> tuple[list[str], list[tuple[Any, ...]], int]:
    """Run ``sql``; interrupt it from a timer thread after ``timeout_s``. Returns (columns, rows, total)."""
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    try:
        cur = con.execute(f"SELECT * FROM ({sql}) AS _q LIMIT {max_rows + 1}")
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchall()
        total = len(rows)
        if total > max_rows:
            total = con.execute(f"SELECT COUNT(*) FROM ({sql}) AS _q").fetchone()[0]  # type: ignore[index]
            rows = rows[:max_rows]
        return cols, rows, int(total)
    except duckdb.InterruptException as e:
        raise QueryTimeout(f"query exceeded {timeout_s:.0f} s and was stopped") from e
    finally:
        timer.cancel()


def explain_cardinality(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    rows = con.execute(f"EXPLAIN (FORMAT json) {sql}").fetchall()
    plan = json.loads(rows[0][1])
    node = plan[0] if isinstance(plan, list) else plan

    def card(n: dict[str, Any]) -> int | None:
        v = n.get("extra_info", {}).get("Estimated Cardinality")
        if v is not None:
            try:
                return int(str(v).replace("~", "").strip())
            except ValueError:
                return None
        for ch in n.get("children", []):
            c = card(ch)
            if c is not None:
                return c
        return None

    return card(node) or 0


def cost_check(con: duckdb.DuckDBPyConnection, sql: str, rt: SqlRuntime) -> GuardResult | None:
    """Reject large unbounded result sets (no LIMIT and no aggregation above the row threshold)."""
    tree = sqlglot.parse_one(sql, read="duckdb")
    est = explain_cardinality(con, sql)
    if est > rt.explain_row_threshold and not _has_limit(tree) and not _is_aggregate_query(tree):
        return _reject(
            "too_many_rows",
            f"The query would return about {est:,} rows (limit "
            f"{rt.explain_row_threshold:,} without LIMIT or aggregation). "
            "Aggregate the data or add a LIMIT.",
        )
    return None


# --------------------------------------------------------------------------- output PII scan
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?880[\s-]?1[3-9]\d{2}[\s-]?\d{6}|01[3-9]\d{2}[\s-]?\d{6})(?!\d)")


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
