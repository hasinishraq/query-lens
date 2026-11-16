"""Order-independent result fingerprints (design doc 9.2), computed in Python over MySQL results.

A fingerprint ignores row order, column order and column names, rounds numbers, lower-cases text, treats a
timestamp at midnight like the same date, and counts duplicate rows (row hashes are summed, not XOR-ed).
Results are streamed from MySQL, so large results are never held in memory as a whole.

:func:`multi_fingerprint` returns three fingerprints at the same precision with rounding phases shifted by
1/3 of a step; two results match if any phase agrees, which absorbs float-sum noise that lands exactly on a
rounding boundary while keeping the tolerance at about 10^-digits.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import pymysql
import pymysql.cursors

PHASES: tuple[tuple[str, float], ...] = (("d", 0.0), ("h", 1 / 3), ("t", 2 / 3))
MAX_ROWS = 2_000_000


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float, Decimal)):
        f = float(v)
        return None if math.isnan(f) else f
    return None


def _norm(v: Any, digits: int, shift: float = 0.0) -> str:
    if v is None:
        return "null"
    f = _num(v)
    if f is not None:
        r = round(f + shift * 10.0**-digits, digits)
        return repr(0.0 if r == 0 else r)
    if isinstance(v, datetime):
        if (v.hour, v.minute, v.second, v.microsecond) == (0, 0, 0, 0):
            return v.strftime("%Y-%m-%d")
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (time, timedelta)):
        return str(v)
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "replace")
    s = str(v).strip().lower()
    try:
        r = round(float(s.replace(",", "")) + shift * 10.0**-digits, digits)
        return repr(0.0 if r == 0 else r)
    except ValueError:
        return s


def _row_hash(row: Iterable[Any], digits: int, shift: float) -> int:
    key = "\x1f".join(sorted(_norm(v, digits, shift) for v in row))
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")


def rows_multi_fingerprint(rows: Iterable[Any], digits: int = 4) -> dict[str, list[Any]]:
    n = 0
    sums = {name: 0 for name, _ in PHASES}
    for r in rows:
        n += 1
        vals = list(r) if isinstance(r, (list, tuple)) else [r]
        for name, shift in PHASES:
            sums[name] = (sums[name] + _row_hash(vals, digits, shift)) % (1 << 64)
    return {name: [n, str(h)] for name, h in sums.items()}


def multi_fingerprint(con: pymysql.connections.Connection, sql: str, digits: int = 4) -> dict[str, list[Any]]:
    """Fingerprints of a query result, streamed from MySQL."""
    cur = con.cursor(pymysql.cursors.SSCursor)
    try:
        cur.execute(sql)

        def gen() -> Iterator[tuple[Any, ...]]:
            for k, r in enumerate(cur):
                if k >= MAX_ROWS:
                    raise ValueError(f"result has more than {MAX_ROWS:,} rows; too large to fingerprint")
                yield r

        return rows_multi_fingerprint(gen(), digits)
    finally:
        cur.close()


def sql_fingerprint(
    con: pymysql.connections.Connection, sql: str, digits: int = 4
) -> tuple[int, str, list[str]]:
    """(row count, content hash, column type codes) for a single rounding phase."""
    with con.cursor() as cur:
        cur.execute(sql)
        types = sorted(str(d[1]) for d in cur.description or [])
        rows = cur.fetchall()
    fp = rows_multi_fingerprint(rows, digits)["d"]
    return int(fp[0]), str(fp[1]), types


def fp_match(a: dict[str, list[Any]] | None, b: dict[str, list[Any]] | None) -> bool:
    if not a or not b:
        return False
    return any(k in b and list(a[k]) == list(b[k]) for k in a)


def rows_fingerprint(rows: list[list[Any]] | list[tuple[Any, ...]], digits: int = 4) -> tuple[int, str]:
    """Single-phase fingerprint for in-memory rows (row count + multiset hash)."""
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
            f = _num(v)
            if f is not None:
                out.append(("n", f))
            elif v is None:
                out.append(("s", "none"))
            else:
                out.append(("s", _norm(v, 6)))
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
