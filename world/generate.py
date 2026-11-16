"""BazaarDW generator CLI.

    python -m bazaar_world.generate --size small --seed 42 --out data/
    python -m bazaar_world.generate --sf 1 --seed 42 --out data/

Builds every table in dependency order, applies planted facts and injected rows, writes
``data/bazaar_<size>.duckdb`` (+ Parquet per table), records table hashes in ``world/table_hashes.json``
and the planted-fact values in ``world/facts.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import duckdb

from .core import WORLD_DIR, World, load_params
from .plant import evaluate_facts, inject_untrusted_text, write_facts
from .roles import build_role_dbs
from .schema import SF_TO_SIZE, TABLES, total_target
from .tables import catalog, customers, logistics, marketing, orders, payments, reference, rest, sellers

log = logging.getLogger("bazaar_world")
HASHES_PATH = WORLD_DIR / "table_hashes.json"


def build_world(size: str, seed: int = 42) -> tuple[World, dict[str, list[int]]]:
    params = load_params()
    w = World(seed=seed, size=size, params=params)
    reference.build(w)
    marketing.build(w)
    customers.build(w)
    catalog.build_categories(w)
    sellers.build_sellers(w)
    sellers.build_tier_history(w)
    catalog.build(w)
    logistics.build_couriers(w)
    logistics.build_warehouses(w)
    orders.build_orders_and_items(w)
    logistics.build_shipments(w)
    payments.build_payments(w)
    payments.build_returns_refunds(w)
    orders.build_status_events(w)
    logistics.build_shipment_events(w)
    sellers.build_payouts(w)
    rest.build_loyalty(w)
    rest.build_inventory(w)
    rest.build_web_sessions(w)
    rest.build_support(w)
    rest.build_reviews(w)
    traps = inject_untrusted_text(w)
    return w, traps


def _select_list(cols: tuple[tuple[str, str], ...]) -> str:
    return ", ".join(f'CAST("{c}" AS {t}) AS "{c}"' for c, t in cols)


def write_duckdb(w: World, path: Path, parquet_dir: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    try:
        for t in TABLES:
            df = w.tables[t.name]
            missing = [c for c in t.column_names if c not in df.columns]
            if missing:
                raise RuntimeError(f"{t.name}: missing columns {missing}")
            frame = df[t.column_names]
            ddl = ", ".join(f'"{c}" {typ}' for c, typ in t.columns)
            key = ", ".join(f'"{k}"' for k in t.key)
            con.execute(f'CREATE TABLE "{t.name}" ({ddl}, PRIMARY KEY ({key}))')
            con.register("_frame", frame)
            con.execute(f'INSERT INTO "{t.name}" SELECT {_select_list(t.columns)} FROM _frame')
            con.unregister("_frame")
            if parquet_dir is not None:
                parquet_dir.mkdir(parents=True, exist_ok=True)
                target = (parquet_dir / f"{t.name}.parquet").as_posix()
                con.execute(f"COPY \"{t.name}\" TO '{target}' (FORMAT parquet)")
        con.execute((WORLD_DIR / "views.sql").read_text(encoding="utf-8"))
        con.execute("CHECKPOINT")
    finally:
        con.close()


def table_hashes(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, Any]]:
    """Row count and order-independent content hash per table."""
    out: dict[str, dict[str, Any]] = {}
    for t in TABLES:
        n, h = con.execute(f'SELECT COUNT(*), SUM(HASH(t)::HUGEINT) FROM "{t.name}" t').fetchone()  # type: ignore[misc]
        out[t.name] = {"rows": int(n), "hash": str(h)}
    return out


def build(size: str, seed: int, out: Path, write_metadata: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    w, traps = build_world(size, seed)
    t_gen = time.perf_counter() - t0
    db_path = out / f"bazaar_{size}.duckdb"
    write_duckdb(w, db_path, out / "parquet" / size)
    build_role_dbs(db_path, size)
    t_total = time.perf_counter() - t0
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        hashes = table_hashes(con)
        facts = evaluate_facts(con, size)
    finally:
        con.close()
    total = sum(v["rows"] for v in hashes.values())
    summary = {
        "size": size,
        "seed": seed,
        "total_rows": total,
        "target_rows": total_target(size),
        "generate_s": round(t_gen, 2),
        "build_s": round(t_total, 2),
        "db_mb": round(db_path.stat().st_size / 1e6, 2),
        "facts": {k: v["holds"] for k, v in facts.items()},
    }
    if write_metadata:
        allh: dict[str, Any] = {}
        if HASHES_PATH.exists():
            allh = json.loads(HASHES_PATH.read_text(encoding="utf-8"))
        allh[f"{size}_seed{seed}"] = {"total_rows": total, "tables": hashes}
        HASHES_PATH.write_text(json.dumps(allh, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_facts(size, seed, facts, traps)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the BazaarDW warehouse")
    ap.add_argument("--size", choices=["small", "full"], default=None)
    ap.add_argument("--sf", default=None, help="scale factor alias: 0.1 (small) or 1 (full)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--no-metadata", action="store_true", help="do not update table_hashes.json / facts.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    size = args.size or SF_TO_SIZE.get(str(args.sf), "small")
    summary = build(size, args.seed, args.out, write_metadata=not args.no_metadata)
    print(json.dumps(summary, indent=2))
    if not all(summary["facts"].values()):
        raise SystemExit("some planted facts do not hold: " + json.dumps(summary["facts"]))


if __name__ == "__main__":
    main()
