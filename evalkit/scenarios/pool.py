"""Dimension sampler: 1,200 scenarios (questions only) for error analysis (Phase 4).

Scenarios come from the golden templates (with more parameter draws), the ambiguous / unanswerable / attack
generators, and persona rewrites (terse, vague, non-native, Banglish). Any question that is identical to a
**test** item is excluded, so error analysis never looks at the test split (rule 3).

    python -m evalkit.scenarios.pool
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bazaar_world.mysqldb import admin_connect, db_name

from ..datasets import DATASETS, load_all
from .build import SEED, make_ctx
from .golden_templates import REGISTRY
from .other_sets import adversarial, ambiguous, unanswerable

ROOT = Path(__file__).resolve().parents[2]
N_SCENARIOS = 1200
SHARE = {"golden_like": 0.70, "ambiguous": 0.10, "unanswerable": 0.08, "attack": 0.12}


def persona(rng: np.random.Generator, q: str, kind: str, bl: str | None = None) -> str:
    if kind == "terse":
        words = q.rstrip("?.").split()
        drop = {"What", "was", "were", "the", "How", "many", "did", "Which", "is", "are", "Show", "in"}
        return " ".join(w for w in words if w not in drop) + "?"
    if kind == "vague":
        return "Quick one: " + q[0].lower() + q[1:] + " Roughly is fine."
    if kind == "non_native":
        return "Please tell me " + q[0].lower() + q[1:].replace("?", "") + " kindly."
    if kind == "banglish" and bl:
        return bl
    return q


def build_pool() -> list[dict]:
    con = admin_connect(db_name("full"))
    small = admin_connect(db_name("small"))
    try:
        ctx = make_ctx(con, small)
    finally:
        con.close()
        small.close()
    test_q = {i.question.strip().lower() for i in load_all() if i.split == "test"}
    rng = np.random.default_rng(SEED + 99)
    out: list[dict] = []
    seen: set[str] = set()

    def add(q: str, dims: dict) -> None:
        k = q.strip().lower()
        if k in seen or k in test_q:
            return
        seen.add(k)
        out.append({"id": f"scn-{len(out) + 1:04d}", "question": q, "dims": dims})

    n_golden = int(N_SCENARIOS * SHARE["golden_like"])
    personas = ["terse", "vague", "non_native", "banglish", "none"]
    per_t = max(4, n_golden // len(REGISTRY) + 2)
    for t in REGISTRY:
        ctx.rng = np.random.default_rng(SEED + 1000 + sum(map(ord, t.id)))
        for inst in t.gen(ctx, max(t.max_n, per_t)):
            p = personas[int(rng.integers(0, len(personas)))]
            add(
                persona(rng, inst.en, p, inst.bl),
                {
                    "intent": t.intent,
                    "difficulty": t.difficulty,
                    "trap": t.trap,
                    "metric": t.metric,
                    "domains": t.domains,
                    "persona": p,
                    "template": t.id,
                },
            )
            if len(out) >= n_golden:
                break
        if len(out) >= n_golden:
            break
    for p in ambiguous():
        add(p.question, {"intent": "clarify", "ambiguity": p.ambiguity, "domains": p.domains})
    for p in unanswerable():
        add(p.question, {"intent": "abstain", "answerable": False, "domains": p.domains})
    for kind, plains in adversarial(list(range(1, 200, 5))).items():
        for p in plains:
            add(p.question, {"intent": "attack", "adversarial": kind, "domains": p.domains})
    # top up with persona variants of golden-like scenarios until N
    base = [s for s in out if "template" in s["dims"]]
    k = 0
    while len(out) < N_SCENARIOS and k < 20 * N_SCENARIOS:
        s = base[k % len(base)]
        p = personas[(k // len(base)) % 3]
        add(persona(rng, s["question"], p), {**s["dims"], "persona": p})
        k += 1
    return out[:N_SCENARIOS]


def main() -> None:
    pool = build_pool()
    path = DATASETS / "scenarios.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s in pool:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps({"scenarios": len(pool)}))  # noqa: T201


if __name__ == "__main__":
    main()
