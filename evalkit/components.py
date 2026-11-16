"""Component metrics (CLAUDE.md 8.7): schema recall, table-set match, latency, scan cost, steps, tokens."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import duckdb
import numpy as np

from .checks import tables_in_sql


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    g = {x.lower() for x in gold}
    if not g:
        return 1.0
    return len(g & {x.lower() for x in retrieved[:k]}) / len(g)


def table_set_match(final_sql: str, gold: list[str]) -> bool:
    return bool(final_sql) and tables_in_sql(final_sql) == {g.lower() for g in gold}


def percentile(values: Iterable[float], q: float) -> float:
    v = list(values)
    return float(np.percentile(v, q)) if v else float("nan")


def rows_scanned(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Rows read by table scans, from DuckDB's EXPLAIN ANALYZE profile."""
    try:
        rows = con.execute(f"EXPLAIN (ANALYZE, FORMAT json) {sql}").fetchall()
    except duckdb.Error:
        return -1
    plan = json.loads(rows[0][1])
    total = 0

    def walk(n: Any) -> None:
        nonlocal total
        if isinstance(n, list):
            for x in n:
                walk(x)
            return
        if not isinstance(n, dict):
            return
        name = str(n.get("operator_name") or n.get("operator_type") or n.get("name") or "")
        if "SCAN" in name.upper():
            total += int(n.get("operator_rows_scanned") or n.get("operator_cardinality") or 0)
        for ch in n.get("children", []):
            walk(ch)

    walk(plan)
    return total


def summarize_components(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate component metrics over item results (dicts from items.jsonl)."""
    out: dict[str, Any] = {}
    rec: dict[int, list[float]] = {5: [], 10: [], 20: []}
    rec_dom: dict[str, list[float]] = defaultdict(list)
    tsm: list[bool] = []
    lat: dict[str, list[float]] = defaultdict(list)
    steps, tok_in, tok_out, cost, llm = [], [], [], [], []
    for it in items:
        gold = it.get("gold_tables") or []
        if it.get("expected_type") == "result" and gold and it.get("first_search_tables") is not None:
            for k in rec:
                rec[k].append(recall_at_k(it["first_search_tables"], gold, k))
            for d in it.get("dims", {}).get("domains", []):
                rec_dom[d].append(recall_at_k(it["first_search_tables"], gold, 10))
            tsm.append(set(it.get("final_tables") or []) == {g.lower() for g in gold})
        lat[it.get("dims", {}).get("difficulty", "?")].append(float(it.get("latency_s", 0)))
        steps.append(it.get("tool_calls", 0))
        llm.append(it.get("llm_calls", 0))
        tok_in.append(it.get("tokens_in", 0))
        tok_out.append(it.get("tokens_out", 0))
        cost.append(it.get("shadow_cost_usd", 0.0))
    out["schema_recall"] = {f"@{k}": (float(np.mean(v)) if v else None) for k, v in rec.items()}
    out["schema_recall_at10_by_domain"] = {d: float(np.mean(v)) for d, v in sorted(rec_dom.items())}
    out["table_set_match"] = float(np.mean(tsm)) if tsm else None
    out["latency"] = {
        d: {"p50": percentile(v, 50), "p95": percentile(v, 95), "n": len(v)} for d, v in sorted(lat.items())
    }
    all_lat = [x for v in lat.values() for x in v]
    out["latency_all"] = {"p50": percentile(all_lat, 50), "p95": percentile(all_lat, 95)}
    out["tool_steps_mean"] = float(np.mean(steps)) if steps else None
    out["llm_calls_mean"] = float(np.mean(llm)) if llm else None
    out["tokens_in_mean"] = float(np.mean(tok_in)) if tok_in else None
    out["tokens_out_mean"] = float(np.mean(tok_out)) if tok_out else None
    out["shadow_cost_usd_total"] = float(np.sum(cost)) if cost else 0.0
    return out
