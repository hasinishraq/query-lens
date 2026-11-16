"""Build ``world/dictionary.yaml`` from ``dictionary_src.py`` and a built database.

python -m bazaar_world.dictionary_build --size small
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import yaml

from .core import WORLD_DIR
from .dictionary_src import COL_DOCS, DOMAIN_SYNONYMS, TABLE_DOCS
from .roles import ROLES, role_columns
from .schema import FOREIGN_KEYS, TABLE_BY_NAME, TABLES

DICT_PATH = WORLD_DIR / "dictionary.yaml"
PII_EXAMPLE = {
    "full_name": "<name>",
    "email": "<email>",
    "phone": "<+8801XXXXXXXXX>",
    "street_address": "<street>",
    "postcode": "<4 digits>",
    "contact_email": "<email>",
    "contact_phone": "<+8801XXXXXXXXX>",
}


def _examples(con: duckdb.DuckDBPyConnection, table: str, col: str) -> list[Any]:
    rows = con.execute(
        f'SELECT DISTINCT CAST("{col}" AS VARCHAR) FROM "{table}" WHERE "{col}" IS NOT NULL ORDER BY 1 LIMIT 3'
    ).fetchall()
    return [r[0][:60] for r in rows]


def build_dictionary(db_path: Path) -> dict[str, Any]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        views = {role: role_columns(con, schema) for role, schema in ROLES.items()}
        out: dict[str, Any] = {"version": 1, "tables": {}}
        for t in TABLES:
            desc, notes = TABLE_DOCS[t.name]
            role_list = [r for r in ROLES if t.name in views[r]]
            cols: dict[str, Any] = {}
            for c, typ in t.columns:
                entry: dict[str, Any] = {
                    "description": COL_DOCS[t.name][c],
                    "type": typ,
                    "examples": [PII_EXAMPLE[c]]
                    if c in t.pii
                    else (["<untrusted text>"] if c in t.untrusted else _examples(con, t.name, c)),
                    "views": [r for r in ROLES if any(vc == c for vc, _ in views[r].get(t.name, []))],
                }
                if c in t.pii:
                    entry["pii"] = True
                if c in t.untrusted:
                    entry["untrusted"] = True
                fk = FOREIGN_KEYS.get((t.name, c))
                if fk:
                    entry["references"] = f"{fk[0]}.{fk[1]}"
                cols[c] = entry
            # derived columns that exist only in role views
            for role in ROLES:
                for vc, vtyp in views[role].get(t.name, []):
                    if vc not in cols:
                        cols[vc] = {
                            "description": COL_DOCS[t.name][vc],
                            "type": vtyp,
                            "examples": [],
                            "views": [role],
                            "derived": True,
                        }
            joins = [
                f"{t.name}.{c} = {rt}.{rc}" for (tt, c), (rt, rc) in FOREIGN_KEYS.items() if tt == t.name
            ]
            joins += [
                f"{tt}.{c} = {t.name}.{rc}"
                for (tt, c), (rt, rc) in FOREIGN_KEYS.items()
                if rt == t.name and tt != t.name
            ]
            n = con.execute(f'SELECT COUNT(*) FROM "{t.name}"').fetchone()
            out["tables"][t.name] = {
                "description": desc,
                "domain": t.domain,
                "synonyms": list(DOMAIN_SYNONYMS.get(t.domain, [])),
                "primary_key": list(t.key),
                "row_count_full": TABLE_BY_NAME[t.name].full_rows,
                "row_count_built": int(n[0]) if n else 0,
                "notes": notes,
                "joins": joins,
                "views": role_list,
                "columns": cols,
            }
        return out
    finally:
        con.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--data", type=Path, default=Path("data"))
    a = ap.parse_args(argv)
    d = build_dictionary(a.data / f"bazaar_{a.size}.duckdb")
    for t in d["tables"].values():
        t.pop("row_count_built", None)
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        f.write(
            "# BazaarDW data dictionary. Generated from world/dictionary_src.py + the built DB by\n"
            "# `python -m bazaar_world.dictionary_build`. Validated by `make dictionary`.\n"
        )
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True, width=120)


if __name__ == "__main__":
    main()
