"""Explicit DuckDB -> MySQL 8 rewrites for the eval templates (docs/decisions/0002-mysql.md).

Templates were first written in DuckDB SQL. Rather than trusting a generic transpiler (sqlglot silently changed
the meaning of date subtraction and quarter truncation), each construct the templates use has one rule here.
Every rewritten gold, alt and trap query was checked against the DuckDB-era results on identical data.

Rules (function calls are parsed with balanced parentheses):
- date_trunc('month', x)    -> CAST(DATE_FORMAT(x, '%Y-%m-01') AS DATE)
- date_trunc('quarter', x)  -> CAST(MAKEDATE(YEAR(x), 1) + INTERVAL (QUARTER(x) - 1) QUARTER AS DATE)
- strftime(x, fmt)          -> DATE_FORMAT(x, fmt)
- date_diff('day', a, b)    -> DATEDIFF(b, a);   date_diff('second', a, b) -> TIMESTAMPDIFF(SECOND, a, b)
- epoch(x)                  -> UNIX_TIMESTAMP(x)        (sessions run with time_zone '+00:00')
- split_part(x, d, 1)       -> SUBSTRING_INDEX(x, d, 1)
- dayofweek(x)              -> (DAYOFWEEK(x) - 1)       (DuckDB counts Sunday = 0, MySQL Sunday = 1)
- AGG(...) FILTER (WHERE c) -> AGG(CASE WHEN c THEN ... END)
- CAST(x AS INTEGER)        -> CAST(x AS SIGNED)
- TIMESTAMP 'YYYY-MM-DD'    -> TIMESTAMP 'YYYY-MM-DD 00:00:00';  INTERVAL '6 hours' -> INTERVAL 6 HOUR
"""

from __future__ import annotations

import re
from collections.abc import Callable


def _match_paren(s: str, open_idx: int) -> int:
    depth = 0
    quote = False
    for i in range(open_idx, len(s)):
        ch = s[i]
        if ch == "'":
            quote = not quote
        elif not quote and ch == "(":
            depth += 1
        elif not quote and ch == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced parentheses in: {s[open_idx : open_idx + 80]}")


def _split_args(inner: str) -> list[str]:
    args, depth, quote, cur = [], 0, False, []
    for ch in inner:
        if ch == "'":
            quote = not quote
        if not quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                args.append("".join(cur).strip())
                cur = []
                continue
        cur.append(ch)
    args.append("".join(cur).strip())
    return args


def _replace_calls(sql: str, name: str, build: Callable[[list[str]], str]) -> str:
    pat = re.compile(rf"(?<![\w.]){name}\s*\(")  # case-sensitive: rewrites emit uppercase MySQL names
    while True:
        m = pat.search(sql)
        if not m:
            return sql
        open_idx = m.end() - 1
        close = _match_paren(sql, open_idx)
        args = _split_args(sql[open_idx + 1 : close])
        sql = sql[: m.start()] + build(args) + sql[close + 1 :]


def _date_trunc(a: list[str]) -> str:
    unit, x = a[0].strip("'").lower(), a[1]
    if unit == "month":
        return f"CAST(DATE_FORMAT({x}, '%Y-%m-01') AS DATE)"
    if unit == "quarter":
        return f"CAST(MAKEDATE(YEAR({x}), 1) + INTERVAL (QUARTER({x}) - 1) QUARTER AS DATE)"
    if unit == "year":
        return f"MAKEDATE(YEAR({x}), 1)"
    raise ValueError(f"date_trunc unit {unit}")


def _date_diff(a: list[str]) -> str:
    unit = a[0].strip("'").lower()
    if unit == "day":
        return f"DATEDIFF({a[2]}, {a[1]})"
    return f"TIMESTAMPDIFF({unit.upper()}, {a[1]}, {a[2]})"


_FILTER = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.I)


def _rewrite_filter(sql: str) -> str:
    while True:
        changed = False
        for m in _FILTER.finditer(sql):
            open_idx = m.end() - 1
            close = _match_paren(sql, open_idx)
            rest = sql[close + 1 :]
            f = re.match(r"\s*FILTER\s*\(\s*WHERE\s", rest, re.I)
            if not f:
                continue
            f_open = close + 1 + rest.index("(")
            f_close = _match_paren(sql, f_open)
            cond = sql[f_open + 1 : f_close].strip()[len("WHERE") :].strip()
            agg, arg = m.group(1).upper(), sql[open_idx + 1 : close].strip()
            if agg == "COUNT" and arg == "*":
                new = f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END)"
            else:
                new = f"{agg}(CASE WHEN {cond} THEN {arg} END)"
            sql = sql[: m.start()] + new + sql[f_close + 1 :]
            changed = True
            break
        if not changed:
            return sql


def to_mysql(sql: str | None) -> str | None:
    if sql is None:
        return None
    s = sql
    s = _replace_calls(s, "date_trunc", _date_trunc)
    s = _replace_calls(s, "strftime", lambda a: f"DATE_FORMAT({a[0]}, {a[1]})")
    s = _replace_calls(s, "date_diff", _date_diff)
    s = _replace_calls(s, "epoch", lambda a: f"UNIX_TIMESTAMP({a[0]})")
    s = _replace_calls(s, "split_part", lambda a: f"SUBSTRING_INDEX({a[0]}, {a[1]}, {a[2]})")
    s = _replace_calls(s, "dayofweek", lambda a: f"(DAYOFWEEK({a[0]}) - 1)")
    s = _rewrite_filter(s)
    s = re.sub(r"\bAS INTEGER\)", "AS SIGNED)", s, flags=re.I)
    s = re.sub(r"TIMESTAMP '(\d{4}-\d{2}-\d{2})'", r"TIMESTAMP '\1 00:00:00'", s)
    s = re.sub(r"INTERVAL '(\d+) hours?'", r"INTERVAL \1 HOUR", s, flags=re.I)
    return s
