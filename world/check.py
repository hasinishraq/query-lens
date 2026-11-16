"""``make world-check``: row counts, totals, table hashes and planted facts of a built warehouse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

from .generate import HASHES_PATH, table_hashes
from .plant import evaluate_facts
from .schema import TABLES, target_rows

ROW_TOLERANCE = 0.02
FULL_MAX_ROWS = 1_000_000
SMALL_MIN_ROWS = 100_000
EXPECTED_TABLES = 35
EXPECTED_COLUMNS = 266


def check(db_path: Path, size: str, seed: int = 42) -> list[str]:
    errors: list[str] = []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        n_tables, n_cols = con.execute(
            "SELECT COUNT(DISTINCT table_name), COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = 'main'"
        ).fetchone()  # type: ignore[misc]
        if n_tables != EXPECTED_TABLES or n_cols != EXPECTED_COLUMNS:
            errors.append(
                f"expected {EXPECTED_TABLES} tables / {EXPECTED_COLUMNS} columns, got {n_tables}/{n_cols}"
            )
        hashes = table_hashes(con)
        total = 0
        for t in TABLES:
            n = hashes[t.name]["rows"]
            total += n
            target = target_rows(t.name, size)
            if t.fixed and n != target:
                errors.append(f"{t.name}: {n} rows, fixed table must have {target}")
            elif abs(n - target) > ROW_TOLERANCE * target:
                errors.append(f"{t.name}: {n} rows, target {target} (+/-2%)")
        if size == "full" and total > FULL_MAX_ROWS:
            errors.append(f"full size has {total} rows (> {FULL_MAX_ROWS})")
        if size == "small" and total < SMALL_MIN_ROWS:
            errors.append(f"small size has {total} rows (< {SMALL_MIN_ROWS})")
        if HASHES_PATH.exists():
            ref = json.loads(HASHES_PATH.read_text(encoding="utf-8")).get(f"{size}_seed{seed}")
            if ref:
                for t, v in ref["tables"].items():
                    if hashes.get(t) != v:
                        errors.append(f"{t}: hash differs from world/table_hashes.json")
        for fid, res in evaluate_facts(con, size).items():
            if not res["holds"]:
                errors.append(f"planted fact {fid} does not hold: {res}")
        print(json.dumps({"size": size, "tables": n_tables, "columns": n_cols, "total_rows": total}))
    finally:
        con.close()
    return errors


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--data", type=Path, default=Path("data"))
    a = ap.parse_args(argv)
    errs = check(a.data / f"bazaar_{a.size}.duckdb", a.size)
    for e in errs:
        print("ERROR", e)
    print("world-check:", "OK" if not errs else f"{len(errs)} errors")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
