"""BazaarDW generator CLI.

    python -m bazaar_world.generate --size small --seed 42 --out data/
    python -m bazaar_world.generate --sf 1 --seed 42 --out data/

Builds every table in dependency order, applies planted facts and injected rows, loads the MySQL database
``bazaar_<size>`` (+ role databases and users, + Parquet per table in ``data/parquet/<size>``), records table
hashes in ``world/table_hashes.json`` and the planted-fact values in ``world/facts.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymysql
import pymysql.cursors

from .core import WORLD_DIR, World, load_params
from .mysqldb import admin_connect, db_name, fetchall
from .plant import evaluate_facts, inject_untrusted_text, write_facts
from .roles import build_role_dbs
from .schema import EXTRA_INDEXES, FOREIGN_KEYS, SF_TO_SIZE, TABLE_BY_NAME, TABLES, mysql_type, total_target
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


def _py_values(series: pd.Series, logical: str) -> list[Any]:
    """Convert a column to plain Python values that PyMySQL can bind (None for missing)."""
    arr = series.to_numpy()
    if logical in ("TIMESTAMP", "DATE"):
        if not np.issubdtype(arr.dtype, np.datetime64):
            arr = np.array(
                [np.datetime64(x) if x is not None and not pd.isna(x) else np.datetime64("NaT") for x in arr]
            )
        unit = "datetime64[us]" if logical == "TIMESTAMP" else "datetime64[D]"
        return arr.astype(unit).astype(object).tolist()  # NaT -> None
    out: list[Any] = []
    for x in series.tolist():
        if x is None or x is pd.NA or (isinstance(x, float) and math.isnan(x)):
            out.append(None)
        elif logical == "BOOLEAN":
            out.append(int(bool(x)))
        elif logical == "INTEGER":
            out.append(int(x))
        elif logical == "DOUBLE":
            out.append(float(x))
        else:
            out.append(str(x))
    return out


def write_mysql(w: World, size: str, parquet_dir: Path | None = None, batch: int = 5000) -> None:
    """Create ``bazaar_<size>`` in MySQL, load every table, add indexes, export Parquet."""
    base = db_name(size)
    con = admin_connect()
    try:
        with con.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{base}`")
            cur.execute(f"CREATE DATABASE `{base}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_as_cs")
            cur.execute(f"USE `{base}`")
            cur.execute("SET SESSION unique_checks = 0, foreign_key_checks = 0")
        con.autocommit(False)
        for t in TABLES:
            df = w.tables[t.name]
            missing = [c for c in t.column_names if c not in df.columns]
            if missing:
                raise RuntimeError(f"{t.name}: missing columns {missing}")
            ddl = ", ".join(f"`{c}` {mysql_type(c, typ)}" for c, typ in t.columns)
            key = ", ".join(f"`{k}`" for k in t.key)
            cols = [_py_values(df[c], typ) for c, typ in t.columns]
            rows = list(zip(*cols, strict=True))
            with con.cursor() as cur:
                cur.execute(f"CREATE TABLE `{t.name}` ({ddl}, PRIMARY KEY ({key})) ENGINE=InnoDB")
                ins = (
                    f"INSERT INTO `{t.name}` ({', '.join(f'`{c}`' for c in t.column_names)}) "
                    f"VALUES ({', '.join(['%s'] * len(t.columns))})"
                )
                for i in range(0, len(rows), batch):
                    cur.executemany(ins, rows[i : i + batch])
            con.commit()
            if parquet_dir is not None:
                parquet_dir.mkdir(parents=True, exist_ok=True)
                frame = pd.DataFrame({c: v for (c, _), v in zip(t.columns, cols, strict=True)})
                frame.to_parquet(parquet_dir / f"{t.name}.parquet", index=False)
        con.autocommit(True)
        with con.cursor() as cur:
            done: set[tuple[str, str]] = set()
            for tbl, col in [*FOREIGN_KEYS.keys(), *EXTRA_INDEXES]:
                if (tbl, col) in done or TABLE_BY_NAME[tbl].key[0] == col:
                    continue
                done.add((tbl, col))
                cur.execute(f"CREATE INDEX `ix_{col}` ON `{tbl}` (`{col}`)")
    finally:
        con.close()


def _norm(v: Any) -> str:
    if v is None:
        return "\\N"
    if isinstance(v, float):
        return repr(round(v, 6))
    if isinstance(v, Decimal):
        return repr(round(float(v), 6))
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v)


def table_hashes(con: Any) -> dict[str, dict[str, Any]]:
    """Row count and order-independent content hash per table (sum of 64-bit row hashes, so duplicates count)."""
    out: dict[str, dict[str, Any]] = {}
    for t in TABLES:
        cur = con.cursor(pymysql.cursors.SSCursor)
        try:
            cur.execute(f"SELECT {', '.join(f'`{c}`' for c in t.column_names)} FROM `{t.name}`")
            n = 0
            total = 0
            for r in cur:
                n += 1
                h = hashlib.blake2b("\x1f".join(_norm(v) for v in r).encode("utf-8"), digest_size=8).digest()
                total = (total + int.from_bytes(h, "little")) % (1 << 64)
        finally:
            cur.close()
        out[t.name] = {"rows": n, "hash": str(total)}
    return out


def build(size: str, seed: int, out: Path, write_metadata: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    w, traps = build_world(size, seed)
    t_gen = time.perf_counter() - t0
    write_mysql(w, size, out / "parquet" / size)
    build_role_dbs(size)
    t_total = time.perf_counter() - t0
    con = admin_connect(db_name(size))
    try:
        hashes = table_hashes(con)
        facts = evaluate_facts(con, size)
        mb = fetchall(
            con,
            "SELECT ROUND(SUM(data_length + index_length) / 1e6, 2) FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = %s",
            (db_name(size),),
        )[0][0]
    finally:
        con.close()
    total = sum(v["rows"] for v in hashes.values())
    summary = {
        "size": size,
        "seed": seed,
        "database": db_name(size),
        "total_rows": total,
        "target_rows": total_target(size),
        "generate_s": round(t_gen, 2),
        "build_s": round(t_total, 2),
        "db_mb": float(mb or 0),
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
    ap.add_argument("--out", type=Path, default=Path("data"), help="folder for the Parquet export")
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
