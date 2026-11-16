"""Order-independent result fingerprints (design doc 9.2).

A fingerprint ignores row order, column order and column names, rounds floats, lower-cases text and
counts duplicate rows (hashes are summed, not XOR-ed). It is computed in DuckDB so large results never
enter Python. :func:`rows_fingerprint` does the same for rows already in memory (agent answers).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import duckdb

FLOAT_TYPES = ("DOUBLE", "FLOAT", "REAL")


def sql_fingerprint(
    con: duckdb.DuckDBPyConnection, sql: str, digits: int = 4, shift: float = 0.0
) -> tuple[int, str, list[str]]:
    """(row count, content hash, sorted column types) of a query result. ``shift`` offsets numbers by a fraction
    of the rounding step before rounding (used for the boundary-safe second fingerprint)."""
    rel = con.sql(sql)
    cols = rel.columns
    types = [str(t) for t in rel.types]
    vals = []
    for col, typ in zip(cols, types, strict=True):
        c = '"' + col.replace('"', '""') + '"'
        if (
            typ in FLOAT_TYPES
            or typ.startswith("DECIMAL")
            or typ in ("HUGEINT", "BIGINT", "INTEGER", "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER")
        ):
            off = f" + {shift * 10**-digits!r}" if shift else ""
            vals.append(f"CAST(ROUND(CAST({c} AS DOUBLE){off}, {digits}) AS VARCHAR)")
        elif typ.startswith("TIMESTAMP"):
            # a timestamp at midnight compares equal to the same DATE (date_trunc vs CAST AS DATE)
            ts = f"CAST({c} AS TIMESTAMP)"
            vals.append(
                f"CASE WHEN CAST({ts} AS TIME) = TIME '00:00:00' THEN strftime({ts}, '%Y-%m-%d') "
                f"ELSE strftime({ts}, '%Y-%m-%d %H:%M:%S') END"
            )
        elif typ == "DATE":
            vals.append(f"strftime({c}, '%Y-%m-%d')")
        elif typ == "BOOLEAN":
            vals.append(f"CASE WHEN {c} THEN 'true' ELSE 'false' END")
        else:
            vals.append(f"LOWER(TRIM(CAST({c} AS VARCHAR)))")
    row = "list_sort([" + ", ".join(f"COALESCE({v}, 'null')" for v in vals) + "])"
    n, h = con.execute(
        f"SELECT COUNT(*), COALESCE(SUM(HASH({row})::HUGEINT), 0) FROM ({sql}) AS _t"
    ).fetchone()  # type: ignore[misc]
    return int(n), str(h), sorted(types)


def _norm(v: Any, digits: int) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, Decimal)):
        f = float(v)
        if math.isnan(f):
            return "null"
        r = round(f, digits)
        return repr(0.0 if r == 0 else r)
    if isinstance(v, datetime):
        if (v.hour, v.minute, v.second) == (0, 0, 0):
            return v.strftime("%Y-%m-%d")
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, date):
        return v.isoformat()
    s = str(v).strip().lower()
    try:
        f = float(s.replace(",", ""))
        return repr(round(f, digits))
    except ValueError:
        return s


def rows_fingerprint(rows: list[list[Any]] | list[tuple[Any, ...]], digits: int = 4) -> tuple[int, str]:
    """Python-side fingerprint (row count + multiset hash) for in-memory rows."""
    c: Counter[str] = Counter()
    for r in rows:
        key = "\x1f".join(sorted(_norm(v, digits) for v in (r if isinstance(r, (list, tuple)) else [r])))
        c[key] += 1
    h = hashlib.sha256()
    for k in sorted(c):
        h.update(f"{k}\x1e{c[k]}\x1d".encode())
    return sum(c.values()), h.hexdigest()


def result_match(
    pred_rows: list[Any], gold_rows: list[Any], float_tol: float = 0.0005, ordered: bool = False
) -> bool:
    """Execution accuracy on in-memory rows: same multiset of rows (column order ignored), numeric tolerance."""
    if len(pred_rows) != len(gold_rows):
        return False

    def norm_row(r: Any) -> list[Any]:
        vals = list(r) if isinstance(r, (list, tuple)) else [r]
        out: list[Any] = []
        for v in vals:
            if isinstance(v, bool) or v is None:
                out.append(("s", str(v).lower()))
            elif isinstance(v, (int, float, Decimal)):
                out.append(("n", float(v)))
            else:
                out.append(("s", str(v).strip().lower()))
        return sorted(out, key=lambda x: (x[0], x[1] if x[0] == "s" else ""))

    def row_eq(a: list[Any], b: list[Any]) -> bool:
        if len(a) != len(b):
            return False
        nums_a = sorted(x[1] for x in a if x[0] == "n")
        nums_b = sorted(x[1] for x in b if x[0] == "n")
        strs_a = sorted(x[1] for x in a if x[0] == "s")
        strs_b = sorted(x[1] for x in b if x[0] == "s")
        if strs_a != strs_b or len(nums_a) != len(nums_b):
            return False
        return all(
            math.isclose(x, y, rel_tol=float_tol, abs_tol=float_tol)
            for x, y in zip(nums_a, nums_b, strict=True)
        )

    p = [norm_row(r) for r in pred_rows]
    g = [norm_row(r) for r in gold_rows]
    if ordered:
        return all(row_eq(a, b) for a, b in zip(p, g, strict=True))
    used = [False] * len(g)
    for a in p:
        for j, b in enumerate(g):
            if not used[j] and row_eq(a, b):
                used[j] = True
                break
        else:
            return False
    return True


def multi_fingerprint(con: duckdb.DuckDBPyConnection, sql: str, digits: int = 4) -> dict[str, list[Any]]:
    """Three fingerprints at the same precision: plain rounding and rounding after 1/3 and 2/3 step shifts. Results
    match if either agrees, which absorbs float-sum noise that lands exactly on a rounding boundary while keeping
    the tolerance at about 10^-digits."""
    out: dict[str, list[Any]] = {}
    for name, shift in (("d", 0.0), ("h", 1 / 3), ("t", 2 / 3)):
        n, h, _ = sql_fingerprint(con, sql, digits=digits, shift=shift)
        out[name] = [n, h]
    return out


def fp_match(a: dict[str, list[Any]] | None, b: dict[str, list[Any]] | None) -> bool:
    if not a or not b:
        return False
    return any(k in b and list(a[k]) == list(b[k]) for k in a)
