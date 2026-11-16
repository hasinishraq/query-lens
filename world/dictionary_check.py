"""``make dictionary``: fail if any DB column (base or role view) is missing from dictionary.yaml or vice versa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import duckdb
import yaml

from .core import WORLD_DIR
from .roles import ROLES, role_columns


def load_dictionary(path: Path | None = None) -> dict[str, Any]:
    with open(path or WORLD_DIR / "dictionary.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check(db_path: Path, dictionary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        base = role_columns(con, "main")
        tables = dictionary["tables"]
        for t, cols in base.items():
            if t not in tables:
                errors.append(f"table {t} missing from dictionary")
                continue
            dcols = tables[t]["columns"]
            for c, typ in cols:
                if c not in dcols:
                    errors.append(f"column {t}.{c} missing from dictionary")
                elif not dcols[c].get("derived") and dcols[c]["type"].upper() != typ.upper():
                    errors.append(f"type mismatch {t}.{c}: db {typ} vs dictionary {dcols[c]['type']}")
            for c, entry in dcols.items():
                if entry.get("derived"):
                    continue
                if c not in {x for x, _ in cols}:
                    errors.append(f"dictionary column {t}.{c} not in database")
        for t in tables:
            if t not in base:
                errors.append(f"dictionary table {t} not in database")
        for role, schema in ROLES.items():
            for t, cols in role_columns(con, schema).items():
                for c, _ in cols:
                    entry = tables.get(t, {}).get("columns", {}).get(c)
                    if entry is None:
                        errors.append(f"view column {schema}.{t}.{c} missing from dictionary")
                    elif role not in entry.get("views", []):
                        errors.append(f"{t}.{c}: dictionary does not list role {role}")
    finally:
        con.close()
    return errors


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--data", type=Path, default=Path("data"))
    a = ap.parse_args(argv)
    errs = check(a.data / f"bazaar_{a.size}.duckdb", load_dictionary())
    for e in errs:
        print("ERROR", e)
    n_cols = sum(len(t["columns"]) for t in load_dictionary()["tables"].values())
    print(
        f"dictionary: {len(load_dictionary()['tables'])} tables, {n_cols} column entries, {len(errs)} errors"
    )
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
