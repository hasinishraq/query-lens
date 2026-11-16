"""Build the 930 eval items (CLAUDE.md 8.2) from templates.

    python -m evalkit.scenarios.build            # writes datasets/*.jsonl and datasets/gold/*.sql

Template counts are chosen with an integer program so every composition rule holds exactly; items are
then assigned to splits with a balancing pass. All gold SQL is a Claude draft (``verified_by: null``).
Existing human verifications (``verified_by``) are preserved for items whose SQL is unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from bazaar_world.mysqldb import admin_connect, db_name, fetchall

from ..datasets import DATASETS, derive_gold_tables
from ..schemas import DOMAINS, TRAPS, Dims, EvalItem, Expected
from .common import Instance
from .golden_templates import REGISTRY, Ctx, Template
from .other_sets import Plain, adversarial, ambiguous, unanswerable

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SEED = 20261116

INTENT_TARGET = {
    "lookup": 40,
    "aggregation": 70,
    "ranking": 60,
    "trend": 60,
    "comparison": 50,
    "ratio": 60,
    "cohort": 30,
    "funnel": 30,
}
DIFF_TARGET = {"easy": 100, "medium": 160, "hard": 100, "extra_hard": 40}
TRAP_TARGET = {"none": 240, **{t: 20 for t in TRAPS}}
METRIC_ITEMS = 200
MIN_PER_METRIC = 5
MIN_PER_DOMAIN = 30
SPLIT_TARGET = {"train": 120, "dev": 120, "test": 160}
SET_SPLITS = {"ambiguous": (24, 24, 32), "unanswerable": (18, 18, 24)}
ADV_SPLITS = {
    "injection": (16, 24),
    "pii": (16, 24),
    "write": (12, 18),
    "resource": (10, 15),
    "jailbreak": (6, 9),
}


def make_ctx(con: Any, small: Any) -> Ctx:
    q = lambda c, s: [r[0] for r in fetchall(c, s)]  # noqa: E731
    shared = sorted(set(q(con, "SELECT name FROM sellers")) & set(q(small, "SELECT name FROM sellers")))
    return Ctx(
        rng=np.random.default_rng(SEED),
        divisions=q(con, "SELECT name FROM regions ORDER BY region_id"),
        couriers=q(con, "SELECT name FROM couriers ORDER BY courier_id"),
        warehouses=q(con, "SELECT name FROM warehouses ORDER BY warehouse_id"),
        l1_categories=q(con, "SELECT name FROM categories WHERE level = 1 ORDER BY category_id"),
        shared_sellers=shared,
        usd_orders=q(
            con,
            "SELECT order_id FROM orders WHERE currency = 'USD' AND NOT is_deleted ORDER BY order_id LIMIT 40",
        ),
        skus=q(
            con,
            "SELECT sku FROM product_variants v JOIN (SELECT variant_id FROM product_price_history GROUP BY 1 "
            "HAVING COUNT(*) >= 3) h USING (variant_id) ORDER BY sku LIMIT 60",
        ),
        campaigns_acq=q(con, "SELECT name FROM marketing_campaigns WHERE objective = 'acquisition'"),
        traffic_sources=q(con, "SELECT DISTINCT traffic_source FROM web_sessions ORDER BY 1"),
        tier_changers=sorted(
            set(
                q(
                    con,
                    "SELECT s.name FROM sellers s JOIN seller_tier_history t USING (seller_id) GROUP BY 1 "
                    "HAVING COUNT(*) > 1",
                )
            )
            & set(shared)
        ),
        depts_with_deleted=q(
            con,
            "SELECT DISTINCT SUBSTRING_INDEX(c.path, ' > ', 1) FROM products p JOIN categories c USING (category_id) "
            "WHERE p.is_deleted ORDER BY 1",
        ),
    )


def instantiate(ctx: Ctx) -> dict[str, list[Instance]]:
    out: dict[str, list[Instance]] = {}
    for t in REGISTRY:
        ctx.rng = np.random.default_rng(SEED + sum(map(ord, t.id)))
        insts = t.gen(ctx, t.max_n)
        seen: set[str] = set()
        uniq = []
        for i in insts:
            if i.en not in seen:
                seen.add(i.en)
                uniq.append(i)
        out[t.id] = uniq
    return out


def solve_counts(templates: list[Template], avail: dict[str, int]) -> dict[str, int]:
    """Integer program: template counts that meet every composition rule exactly."""
    n = len(templates)
    nv = 2 * n  # x_t and overflow y_t >= x_t - soft cap
    rows, lo, hi = [], [], []

    def add(coef: dict[int, float], lb: float, ub: float) -> None:
        r = np.zeros(nv)
        for k, v in coef.items():
            r[k] = v
        rows.append(r)
        lo.append(lb)
        hi.append(ub)

    for intent, tgt in INTENT_TARGET.items():
        add({i: 1 for i, t in enumerate(templates) if t.intent == intent}, tgt, tgt)
    for d, tgt in DIFF_TARGET.items():
        add({i: 1 for i, t in enumerate(templates) if t.difficulty == d}, tgt, tgt)
    for tr, tgt in TRAP_TARGET.items():
        add({i: 1 for i, t in enumerate(templates) if t.trap == tr}, tgt, tgt)
    add({i: 1 for i, t in enumerate(templates) if t.metric}, METRIC_ITEMS, METRIC_ITEMS)
    for m in {t.metric for t in templates if t.metric}:
        add({i: 1 for i, t in enumerate(templates) if t.metric == m}, MIN_PER_METRIC, np.inf)
    for d in DOMAINS:
        add({i: 1 for i, t in enumerate(templates) if d in t.domains}, MIN_PER_DOMAIN, np.inf)
    soft = 4
    for i in range(n):
        add({n + i: 1, i: -1}, -soft, np.inf)  # y_i >= x_i - soft
    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    for i, t in enumerate(templates):
        cap = min(t.max_n, avail[t.id])
        lb[i] = min(t.min_n, cap)
        ub[i] = cap
    c = np.concatenate([np.zeros(n), np.ones(n)])
    res = milp(
        c,
        constraints=LinearConstraint(np.array(rows), lo, hi),
        integrality=np.ones(nv),
        bounds=Bounds(lb, ub),
        options={"time_limit": 60},
    )
    if res.x is None:
        raise RuntimeError(f"composition is infeasible with the current templates: {res.message}")
    return {t.id: int(round(res.x[i])) for i, t in enumerate(templates)}


def assign_splits(keys: list[tuple[str, ...]], targets: dict[str, int], seed: int) -> list[str]:
    """Greedy balanced assignment: each item goes to the split whose categories are most under target."""
    total = sum(targets.values())
    frac = {s: v / total for s, v in targets.items()}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    counts: dict[str, Counter[tuple[int, str]]] = {s: Counter() for s in targets}
    marg: list[Counter[tuple[int, str]]] = [Counter() for _ in range(len(keys[0]) if keys else 0)]
    for k in keys:
        for j, v in enumerate(k):
            marg[j][(j, v)] += 1
    size = Counter()
    out = [""] * len(keys)
    for idx in order:
        k = keys[idx]
        best, best_score = None, None
        for s in targets:
            if size[s] >= targets[s]:
                continue
            score = sum(
                (counts[s][(j, v)] + 1) / max(marg[j][(j, v)] * frac[s], 1e-9) for j, v in enumerate(k)
            )
            score += (size[s] + 1) / targets[s]
            if best_score is None or score < best_score:
                best, best_score = s, score
        assert best is not None
        out[idx] = best
        size[best] += 1
        for j, v in enumerate(k):
            counts[best][(j, v)] += 1
    return out


def _existing_verifications() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for f in DATASETS.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("verified_by") and d.get("gold_sql_file"):
                    p = DATASETS / d["gold_sql_file"]
                    out[d["id"]] = (d["verified_by"], p.read_text(encoding="utf-8") if p.exists() else "")
    return out


def build(write: bool = True) -> dict[str, list[EvalItem]]:
    con = admin_connect(db_name("full"))
    small = admin_connect(db_name("small"))
    try:
        ctx = make_ctx(con, small)
        injected = [
            r[0]
            for r in fetchall(
                con,
                "SELECT DISTINCT product_id FROM product_reviews WHERE LOWER(body) LIKE '%%instruction%%' "
                "OR LOWER(body) LIKE '%%ignore%%' OR LOWER(body) LIKE '%%system:%%' OR LOWER(body) LIKE '%%admin%%' "
                "OR LOWER(body) LIKE '%%tool:%%' OR body LIKE '%%নির্দেশনা%%' OR LOWER(body) LIKE '%%read_csv%%' "
                "OR LOWER(body) LIKE '%%revenue was 0%%' OR LOWER(body) LIKE '%%disregard%%' "
                "OR LOWER(body) LIKE '%%street_address%%' OR LOWER(body) LIKE '%%koro ar%%' ORDER BY 1",
                (),
            )
        ]
    finally:
        con.close()
        small.close()
    insts = instantiate(ctx)
    counts = solve_counts(REGISTRY, {k: len(v) for k, v in insts.items()})
    prior = _existing_verifications()
    gold_dir = DATASETS / "gold"
    files: dict[str, str] = {}

    # ---------------------------------------------------------------- golden
    golden_raw: list[tuple[Template, Instance]] = []
    for t in REGISTRY:
        golden_raw += [(t, i) for i in insts[t.id][: counts[t.id]]]
    keys = [(t.intent, i.difficulty or t.difficulty, t.trap) for t, i in golden_raw]
    splits = None
    for attempt in range(200):
        cand = assign_splits(keys, SPLIT_TARGET, SEED + attempt)
        if _splits_ok(keys, cand):
            splits = cand
            break
    if splits is None:
        raise RuntimeError("could not balance golden splits within +/-2")
    golden: list[EvalItem] = []
    for n, ((t, inst), split) in enumerate(zip(golden_raw, splits, strict=True), start=1):
        gid = f"gld-{n:04d}"
        files[f"{gid}.sql"] = inst.gold + "\n"
        files[f"{gid}.alt.sql"] = inst.alt + "\n"
        if inst.trap:
            files[f"{gid}.trap.sql"] = inst.trap + "\n"
        verified = prior.get(gid)
        golden.append(
            EvalItem(
                id=gid,
                split=split,
                set="golden",
                question=inst.en,  # type: ignore[arg-type]
                dims=Dims(
                    intent=t.intent,
                    difficulty=inst.difficulty or t.difficulty,
                    domains=t.domains,
                    trap=t.trap,  # type: ignore[arg-type]
                    metric=t.metric,
                    persona="terse",
                ),
                gold_sql_file=f"gold/{gid}.sql",
                gold_tables=derive_gold_tables(inst.gold),
                expected=Expected(type="result", ordered=inst.ordered, must_mention_any=inst.must_mention),
                planted_fact=inst.planted_fact
                or (t.planted_fact if t.planted_fact and inst.planted_fact else None),
                verified_by=verified[0] if verified and verified[1].strip() == inst.gold.strip() else None,
                notes=f"template {t.id}; params {inst.key}",
            )
        )
    meta = {g.id: inst for g, (_, inst) in zip(golden, golden_raw, strict=True)}

    # ---------------------------------------------------------------- bilingual (pair with dev/test golden)
    rng = np.random.default_rng(SEED + 7)
    bil_bn: list[EvalItem] = []
    bil_bl: list[EvalItem] = []
    for split in ("dev", "test"):
        pool = [g for g in golden if g.split == split]
        chosen = sorted(rng.choice(len(pool), size=60, replace=False))
        for k, idx in enumerate(chosen):
            g = pool[idx]
            inst = meta[g.id]
            for lang, qtext, bucket, prefix, persona in (
                ("bn", inst.bn, bil_bn, "bn", "non_native"),
                ("banglish", inst.bl, bil_bl, "bl", "banglish"),
            ):
                bucket.append(
                    g.model_copy(
                        update={
                            "id": f"{prefix}-{split[0]}{k + 1:03d}",
                            "set": f"bilingual_{lang}",
                            "lang": lang,
                            "question": qtext,
                            "pair_id": g.id,
                            "dims": g.dims.model_copy(update={"persona": persona}),
                            "verified_by": None,
                            "notes": f"pairs with {g.id}",
                        }
                    )
                )

    # ---------------------------------------------------------------- ambiguous / unanswerable
    def plain_items(
        plains: list[Plain],
        set_name: str,
        prefix: str,
        intent: str,
        split_counts: tuple[int, ...],
        answerable: bool,
    ) -> list[EvalItem]:
        assert len(plains) == sum(split_counts), (set_name, len(plains))
        labels = ["train"] * split_counts[0] + ["dev"] * split_counts[1] + ["test"] * split_counts[2]
        order = np.random.default_rng(SEED + len(prefix)).permutation(len(plains))
        out = []
        for n, (pi, lab) in enumerate(zip(order, labels, strict=True), start=1):
            p = plains[pi]
            out.append(
                EvalItem(
                    id=f"{prefix}-{n:04d}",
                    split=lab,
                    set=set_name,
                    question=p.question,  # type: ignore[arg-type]
                    dims=Dims(
                        intent=intent,
                        difficulty="medium",
                        domains=p.domains,
                        ambiguity=p.ambiguity,  # type: ignore[arg-type]
                        answerable=answerable,
                        persona="vague" if set_name == "ambiguous" else "terse",
                    ),
                    expected=Expected(type=p.expected),
                    notes=p.key,
                )
            )  # type: ignore[arg-type]
        return sorted(out, key=lambda x: x.id)

    amb = plain_items(ambiguous(), "ambiguous", "amb", "clarify", SET_SPLITS["ambiguous"], True)
    una = plain_items(unanswerable(), "unanswerable", "una", "abstain", SET_SPLITS["unanswerable"], False)

    # ---------------------------------------------------------------- adversarial (dev / test)
    adv: list[EvalItem] = []
    groups = adversarial(injected)
    if len(groups["injection"]) < 40:  # fewer distinct injected products than rows: add phrasings
        base = list(groups["injection"])
        k = 0
        while len(groups["injection"]) < 40:
            b = base[k % len(base)]
            groups["injection"].append(
                Plain(
                    b.key + f":v{k}",
                    b.question.replace("3 most recent", "2 most recent").replace("LIMIT 3", "LIMIT 2")
                    + " Summarise them.",
                    b.domains,
                    adversarial="injection",
                    expected="result",
                    gold=(b.gold or "").replace("LIMIT 3", "LIMIT 2"),
                    forbidden_columns=b.forbidden_columns,
                )
            )
            k += 1
    n = 0
    for kind, (n_dev, n_test) in ADV_SPLITS.items():
        plains = groups[kind]
        assert len(plains) == n_dev + n_test, (kind, len(plains))
        order = np.random.default_rng(SEED + len(kind)).permutation(len(plains))
        for j, pi in enumerate(order):
            p = plains[pi]
            n += 1
            aid = f"adv-{n:04d}"
            gold_file = None
            tables: list[str] = []
            if p.gold:
                files[f"{aid}.sql"] = p.gold + "\n"
                gold_file = f"gold/{aid}.sql"
                tables = derive_gold_tables(p.gold)
            adv.append(
                EvalItem(
                    id=aid,
                    split="dev" if j < n_dev else "test",
                    set="adversarial",
                    question=p.question,  # type: ignore[arg-type]
                    dims=Dims(
                        intent="attack",
                        difficulty="medium",
                        domains=p.domains,
                        adversarial=kind,  # type: ignore[arg-type]
                        answerable=p.expected == "result",
                    ),
                    gold_sql_file=gold_file,
                    gold_tables=tables,
                    expected=Expected(type=p.expected, forbidden_columns=p.forbidden_columns),  # type: ignore[arg-type]
                    notes=p.key,
                )
            )
    sets = {
        "golden": golden,
        "ambiguous": amb,
        "unanswerable": una,
        "adversarial": adv,
        "bilingual_bn": bil_bn,
        "bilingual_banglish": bil_bl,
    }
    if write:
        gold_dir.mkdir(parents=True, exist_ok=True)
        for old in gold_dir.glob("*.sql"):
            old.unlink()
        for name, body in files.items():
            (gold_dir / name).write_text(body, encoding="utf-8")
        for name, items in sets.items():
            path = DATASETS / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for it in items:
                    f.write(it.model_dump_json(exclude_none=False) + "\n")
        reg = DATASETS / "regression.jsonl"
        if not reg.exists():
            reg.write_text("", encoding="utf-8")
    log.info("template counts: %s", {k: v for k, v in counts.items() if v})
    return sets


def _splits_ok(keys: list[tuple[str, ...]], splits: list[str]) -> bool:
    total = len(keys)
    for j in range(len(keys[0])):
        by: dict[str, Counter[str]] = defaultdict(Counter)
        for k, s in zip(keys, splits, strict=True):
            by[k[j]][s] += 1
        for _v, c in by.items():
            n = sum(c.values())
            for s, tgt in SPLIT_TARGET.items():
                if abs(c[s] - n * tgt / total) > 2:
                    return False
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sets = build()
    print(json.dumps({k: len(v) for k, v in sets.items()}))  # noqa: T201


if __name__ == "__main__":
    main()


__all__: list[Any] = ["build"]
