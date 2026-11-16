"""Load, validate and slice eval datasets; balance report; gold results (CLAUDE.md 8.1-8.3)."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

from .fingerprint import fp_match, multi_fingerprint, sql_fingerprint
from .schemas import DOMAINS, TRAPS, EvalItem

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
SET_FILES = [
    "golden",
    "ambiguous",
    "unanswerable",
    "adversarial",
    "bilingual_bn",
    "bilingual_banglish",
    "regression",
]

COUNTS = {  # set -> (train, dev, test)
    "golden": (120, 120, 160),
    "ambiguous": (24, 24, 32),
    "unanswerable": (18, 18, 24),
    "adversarial": (0, 60, 90),
    "bilingual_bn": (0, 60, 60),
    "bilingual_banglish": (0, 60, 60),
}
INTENTS = {
    "lookup": 40,
    "aggregation": 70,
    "ranking": 60,
    "trend": 60,
    "comparison": 50,
    "ratio": 60,
    "cohort": 30,
    "funnel": 30,
}
DIFFICULTIES = {"easy": 100, "medium": 160, "hard": 100, "extra_hard": 40}
ADV_TYPES = {
    "injection": (16, 24),
    "pii": (16, 24),
    "write": (12, 18),
    "resource": (10, 15),
    "jailbreak": (6, 9),
}
PLANTED = [f"F0{i}" for i in range(1, 9)]
CI_SAMPLE = 60


def derive_gold_tables(sql: str) -> list[str]:
    tree = sqlglot.parse_one(sql, read="duckdb")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return sorted({t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes})


def load_set(name: str) -> list[EvalItem]:
    path = DATASETS / f"{name}.jsonl"
    if not path.exists():
        return []
    return [
        EvalItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_all() -> list[EvalItem]:
    out: list[EvalItem] = []
    for s in SET_FILES:
        out += load_set(s)
    return out


def gold_sql(item: EvalItem) -> str | None:
    if not item.gold_sql_file:
        return None
    return (DATASETS / item.gold_sql_file).read_text(encoding="utf-8").strip()


def variant_sql(item: EvalItem, kind: str) -> str | None:
    """kind = 'alt' or 'trap'."""
    if not item.gold_sql_file:
        return None
    p = DATASETS / item.gold_sql_file.replace(".sql", f".{kind}.sql")
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def dataset_hash(items: list[EvalItem]) -> str:
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x.id):
        h.update(it.model_dump_json().encode())
        g = gold_sql(it)
        if g:
            h.update(g.encode())
    return h.hexdigest()[:16]


def select_slice(slice_name: str, items: list[EvalItem] | None = None) -> list[EvalItem]:
    """Slices: train | dev | test | adv | ci (60 fixed dev golden items) | <set>:<split> | ids:a,b,c."""
    items = items if items is not None else load_all()
    if slice_name.startswith("ids:"):
        ids = set(slice_name[4:].split(","))
        return [i for i in items if i.id in ids]
    if ":" in slice_name:
        s, sp = slice_name.split(":", 1)
        return [i for i in items if i.set == s and i.split == sp]
    if slice_name == "ci":
        dev = sorted((i for i in items if i.set == "golden" and i.split == "dev"), key=lambda x: x.id)
        step = max(1, len(dev) // CI_SAMPLE)
        return dev[::step][:CI_SAMPLE]
    if slice_name == "adv":
        return [i for i in items if i.set == "adversarial"]
    if slice_name == "regression":
        return [i for i in items if i.set == "regression"]
    return [i for i in items if i.split == slice_name and i.set != "regression"]


# --------------------------------------------------------------------------- composition report
def report(items: list[EvalItem] | None = None) -> tuple[dict[str, Any], list[str]]:
    items = items if items is not None else load_all()
    errors: list[str] = []
    by_set: dict[str, list[EvalItem]] = defaultdict(list)
    for it in items:
        by_set[it.set].append(it)
    ids = Counter(i.id for i in items)
    errors += [f"duplicate id {k}" for k, v in ids.items() if v > 1]
    rep: dict[str, Any] = {"sets": {}}
    for s, (tr, dv, te) in COUNTS.items():
        c = Counter(i.split for i in by_set[s])
        rep["sets"][s] = {"train": c["train"], "dev": c["dev"], "test": c["test"], "total": len(by_set[s])}
        if (c["train"], c["dev"], c["test"]) != (tr, dv, te):
            errors.append(f"{s}: splits {c['train']}/{c['dev']}/{c['test']} != {tr}/{dv}/{te}")
    total = sum(len(by_set[s]) for s in COUNTS)
    rep["total"] = total
    if total != 930:
        errors.append(f"total items {total} != 930")
    g = by_set["golden"]
    intents = Counter(i.dims.intent for i in g)
    diffs = Counter(i.dims.difficulty for i in g)
    traps = Counter(i.dims.trap for i in g)
    rep["golden"] = {"intent": dict(intents), "difficulty": dict(diffs), "trap": dict(traps)}
    for k, v in INTENTS.items():
        if intents[k] != v:
            errors.append(f"golden intent {k}: {intents[k]} != {v}")
    for k, v in DIFFICULTIES.items():
        if diffs[k] != v:
            errors.append(f"golden difficulty {k}: {diffs[k]} != {v}")
    if traps["none"] != 240:
        errors.append(f"golden trap none: {traps['none']} != 240")
    for t in TRAPS:
        if traps[t] != 20:
            errors.append(f"golden trap {t}: {traps[t]} != 20")
    dom = Counter(d for i in g for d in set(i.dims.domains))
    rep["golden"]["domains"] = dict(dom)
    for d in DOMAINS:
        if dom[d] < 30:
            errors.append(f"golden domain {d}: {dom[d]} < 30")
    metric_items = [i for i in g if i.dims.metric]
    mc = Counter(i.dims.metric for i in metric_items)
    rep["golden"]["metric_items"] = len(metric_items)
    rep["golden"]["metrics"] = dict(mc)
    if len(metric_items) != 200:
        errors.append(f"golden metric items {len(metric_items)} != 200")
    from querylens.semantic import load_metrics

    for m in load_metrics():
        if mc[m] < 5:
            errors.append(f"metric {m}: {mc[m]} < 5 items")
    pf = Counter(i.planted_fact for i in g if i.planted_fact)
    rep["golden"]["planted_facts"] = dict(pf)
    for f in PLANTED:
        if pf[f] < 3:
            errors.append(f"planted fact {f}: {pf[f]} < 3 items")
    # split proportions within +/-2 for each golden dimension value
    for dim in ("intent", "difficulty", "trap"):
        by: dict[str, Counter[str]] = defaultdict(Counter)
        for i in g:
            by[getattr(i.dims, dim)][i.split] += 1
        for v, c in by.items():
            n = sum(c.values())
            for sp, tgt in zip(("train", "dev", "test"), COUNTS["golden"], strict=True):
                exp_n = n * tgt / 400
                if abs(c[sp] - exp_n) > 2:
                    errors.append(f"golden {dim}={v}: {sp} has {c[sp]}, expected {exp_n:.1f} (+/-2)")
    adv = by_set["adversarial"]
    at = {k: Counter(i.split for i in adv if i.dims.adversarial == k) for k in ADV_TYPES}
    rep["adversarial"] = {k: dict(v) for k, v in at.items()}
    for k, (d, t) in ADV_TYPES.items():
        if (at[k]["dev"], at[k]["test"]) != (d, t):
            errors.append(f"adversarial {k}: dev/test {at[k]['dev']}/{at[k]['test']} != {d}/{t}")
    gold_by_id = {i.id: i for i in g}
    for s in ("bilingual_bn", "bilingual_banglish"):
        for i in by_set[s]:
            pair = gold_by_id.get(i.pair_id or "")
            if pair is None:
                errors.append(f"{i.id}: pair {i.pair_id} not a golden item")
            elif pair.split != i.split:
                errors.append(f"{i.id}: split {i.split} differs from its pair {pair.id} ({pair.split})")
            elif i.gold_sql_file != pair.gold_sql_file:
                errors.append(f"{i.id}: must reuse the gold SQL of {pair.id}")
    for i in items:
        sql = gold_sql(i)
        if sql is not None and derive_gold_tables(sql) != sorted(i.gold_tables):
            errors.append(f"{i.id}: gold_tables out of sync with gold SQL")
    rep["verified"] = {s: sum(1 for i in by_set[s] if i.verified_by) for s in COUNTS}
    return rep, errors


# --------------------------------------------------------------------------- gold results
def build_gold_results(
    snapshot: str, data_dir: Path, items: list[EvalItem] | None = None, check_alt: bool = True
) -> dict[str, Any]:
    """Run every gold SQL on a snapshot (analyst role DB); store Parquet + fingerprints; double-written check."""
    from querylens.paths import db_path

    items = items if items is not None else load_all()
    out_dir = DATASETS / "gold_results" / snapshot
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(data_dir / db_path(snapshot).name), read_only=True)
    hashes: dict[str, Any] = {}
    problems: list[str] = []
    done_files: set[str] = set()
    try:
        for it in items:
            sql = gold_sql(it)
            if sql is None or it.gold_sql_file in done_files:
                continue
            done_files.add(it.gold_sql_file or "")
            key = Path(it.gold_sql_file or "").stem
            try:
                fp = multi_fingerprint(con, sql)
                _, _, types = sql_fingerprint(con, sql)
                con.execute(f"COPY ({sql}) TO '{(out_dir / f'{key}.parquet').as_posix()}' (FORMAT parquet)")
            except duckdb.Error as e:
                problems.append(f"{it.id}: gold SQL fails on {snapshot}: {str(e).splitlines()[0][:200]}")
                continue
            entry: dict[str, Any] = {"rows": fp["d"][0], "fp": fp, "types": types}
            alt = variant_sql(it, "alt")
            if alt and check_alt:
                try:
                    afp = multi_fingerprint(con, alt)
                    entry["alt_match"] = fp_match(afp, fp)
                    if not entry["alt_match"]:
                        problems.append(
                            f"{it.id}: alt SQL fingerprint differs on {snapshot} "
                            f"({afp['d'][0]} vs {fp['d'][0]} rows)"
                        )
                except duckdb.Error as e:
                    entry["alt_match"] = False
                    problems.append(f"{it.id}: alt SQL fails on {snapshot}: {str(e).splitlines()[0][:200]}")
            trap = variant_sql(it, "trap")
            if trap:
                try:
                    tfp = multi_fingerprint(con, trap)
                    entry["trap_fp"] = tfp
                    entry["trap_differs"] = not fp_match(tfp, fp)
                except duckdb.Error as e:
                    problems.append(f"{it.id}: trap SQL fails on {snapshot}: {str(e).splitlines()[0][:200]}")
            hashes[key] = entry
    finally:
        con.close()
    (out_dir / "hashes.json").write_text(
        json.dumps(hashes, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"snapshot": snapshot, "n": len(hashes), "problems": problems}


def load_gold_hashes(snapshot: str) -> dict[str, Any]:
    p = DATASETS / "gold_results" / snapshot / "hashes.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
