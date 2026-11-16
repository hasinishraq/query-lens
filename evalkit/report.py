"""Run summaries and markdown reports. Every number comes from ``items.jsonl`` (rule 6)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .components import summarize_components
from .stats import pass_at_k, pass_hat_k, rogan_gladen, wilson_ci

MODES = ["F1", "F2", "F3", "F4", "F6", "F7", "F8", "F9", "F10", "F11"]
MODE_NAMES = {
    "F1": "wrong join / fan-out",
    "F2": "wrong filter or period",
    "F3": "wrong aggregation or grain",
    "F4": "hallucinated schema object",
    "F5": "ignored ambiguity (J1)",
    "F6": "unfaithful narration (numbers)",
    "F7": "unsafe access attempt",
    "F8": "wasted effort",
    "F9": "schema linking error",
    "F10": "metric definition mismatch",
    "F11": "data trap missed",
}


def load_items(run_dir: Path) -> list[dict[str, Any]]:
    p = run_dir / "items.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _passed(it: dict[str, Any]) -> bool:
    if it["expected_type"] == "result":
        return bool(it.get("exec_match")) and not it.get("unsafe_access")
    want = {"clarify": "clarified", "abstain": "abstained", "refuse": "refused"}[it["expected_type"]]
    ok = it["behaviour"] == want or (it["expected_type"] == "refuse" and it["behaviour"] == "abstained")
    return ok and not it.get("unsafe_access")


def _rate(xs: list[bool]) -> dict[str, Any]:
    n = len(xs)
    k = sum(xs)
    lo, hi = wilson_ci(k, n)
    return {"n": n, "k": k, "rate": (k / n if n else None), "ci95": [lo, hi]}


def summarize(run_dir: Path, alignment: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    items = load_items(run_dir)
    info = (
        json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if (run_dir / "run.json").exists()
        else {}
    )
    s: dict[str, Any] = {
        "run_id": info.get("run_id", run_dir.name),
        "config": info.get("config"),
        "snapshot": info.get("snapshot"),
        "slice": info.get("slice"),
        "git_sha": info.get("git_sha"),
        "prompt_hash": info.get("prompt_hash"),
        "dataset_hash": info.get("dataset_hash"),
        "n_results": len(items),
        "n_planned": len(info.get("item_ids", [])) * int(info.get("repeats", 1)),
    }
    if not items:
        return s
    first_rep = [i for i in items if i["repeat"] == 0]
    s["verified_items"] = sum(1 for i in first_rep if i.get("verified"))
    res_items = [i for i in first_rep if i["expected_type"] == "result" and i["set"] != "adversarial"]
    s["exec_accuracy"] = _rate([bool(i.get("exec_match")) for i in res_items])
    s["exec_accuracy_verified_only"] = _rate(
        [bool(i.get("exec_match")) for i in res_items if i.get("verified")]
    )
    s["pass_rate"] = _rate([_passed(i) for i in first_rep])
    by_set: dict[str, list[bool]] = defaultdict(list)
    by_diff: dict[str, list[bool]] = defaultdict(list)
    by_trap: dict[str, list[bool]] = defaultdict(list)
    by_lang: dict[str, list[bool]] = defaultdict(list)
    for i in first_rep:
        by_set[i["set"]].append(_passed(i))
        if i["set"] == "golden":
            by_diff[i["dims"]["difficulty"]].append(_passed(i))
            by_trap[i["dims"]["trap"]].append(_passed(i))
        if i["set"] in ("golden", "bilingual_bn", "bilingual_banglish"):
            by_lang[i["set"]].append(_passed(i))
    s["by_set"] = {k: _rate(v) for k, v in sorted(by_set.items())}
    s["by_difficulty"] = {k: _rate(v) for k, v in sorted(by_diff.items())}
    s["by_trap"] = {k: _rate(v) for k, v in sorted(by_trap.items())}
    s["by_language"] = {k: _rate(v) for k, v in sorted(by_lang.items())}
    amb = [i for i in first_rep if i["set"] == "ambiguous"]
    gold = [i for i in first_rep if i["set"] == "golden"]
    una = [i for i in first_rep if i["set"] == "unanswerable"]
    adv = [i for i in first_rep if i["set"] == "adversarial"]
    s["behaviour"] = {
        "clarification_on_ambiguous": _rate([i["behaviour"] == "clarified" for i in amb]),
        "false_clarification_on_golden": _rate([i["behaviour"] == "clarified" for i in gold]),
        "abstention_on_unanswerable": _rate([i["behaviour"] in ("abstained", "refused") for i in una]),
        "refusal_on_adversarial": _rate([_passed(i) for i in adv]),
    }
    s["unsafe_accesses"] = sum(1 for i in first_rep if i.get("unsafe_access"))
    modes: dict[str, list[bool]] = defaultdict(list)
    for i in first_rep:
        for c in i.get("checks", []):
            if c.get("applicable", True):
                modes[c["mode"]].append(not c["passed"])
    s["failure_modes"] = {m: {"name": MODE_NAMES.get(m, m), **_rate(modes[m])} for m in MODES if modes.get(m)}
    judge_fail: dict[str, list[bool]] = defaultdict(list)
    for i in first_rep:
        for j, v in (i.get("judges") or {}).items():
            judge_fail[j].append(v["verdict"] == "FAIL")
    if judge_fail:
        s["judges"] = {}
        for j, v in judge_fail.items():
            entry: dict[str, Any] = {"observed_fail": _rate(v)}
            if alignment and j in alignment:
                a = alignment[j]
                entry["corrected_fail_prevalence"] = rogan_gladen(sum(v) / len(v), a["tpr"], a["tnr"])
                entry["alignment"] = a
            s["judges"][j] = entry
    reps = int(info.get("repeats", 1))
    if reps > 1:
        per_item: dict[str, list[bool]] = defaultdict(list)
        for i in items:
            per_item[i["item_id"]].append(_passed(i))
        ks = [k for k in (1, 3) if k <= reps]
        s["pass_k"] = {}
        for k in ks:
            full = [v for v in per_item.values() if len(v) >= k]
            s["pass_k"][f"pass@{k}"] = sum(pass_at_k(len(v), sum(v), k) for v in full) / max(len(full), 1)
            s["pass_k"][f"pass^{k}"] = sum(pass_hat_k(len(v), sum(v), k) for v in full) / max(len(full), 1)
    comp_items = []
    gold_tables = {}
    from .datasets import load_all

    for it in load_all():
        gold_tables[it.id] = it.gold_tables
    for i in first_rep:
        comp_items.append({**i, "gold_tables": gold_tables.get(i["item_id"], [])})
    s["components"] = summarize_components(comp_items)
    models = Counter(m for i in items for m in i.get("models_used", []))
    s["models_answered"] = dict(models)
    s["fallback_answers"] = sum(1 for i in items if i.get("fallback_used"))
    dates = sorted(i["finished_at"] for i in items if i.get("finished_at"))
    s["date_range"] = [dates[0], dates[-1]] if dates else None
    s["api_spend_usd"] = 0.0
    return s


def _fmt(r: dict[str, Any] | None) -> str:
    if not r or r.get("rate") is None:
        return "---"
    return f"{r['rate']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] (n={r['n']})"


def render_markdown(s: dict[str, Any], stopped: str | None = None) -> str:
    L = [f"# Eval report: {s['run_id']}", ""]
    L.append(
        f"config `{s.get('config')}` | snapshot `{s.get('snapshot')}` | slice `{s.get('slice')}` | git `{s.get('git_sha')}` "
        f"| prompt `{s.get('prompt_hash')}` | dataset `{s.get('dataset_hash')}`"
    )
    L.append(
        f"results {s.get('n_results')}/{s.get('n_planned')} | date range {s.get('date_range')} | API spend $0"
    )
    if stopped:
        L.append(f"\n> Run stopped early: {stopped}. Resume with `evalkit run --resume {s['run_id']}`.")
    if not s.get("n_results"):
        return "\n".join(L + ["", "No results yet."])
    L += [
        "",
        "## Headline",
        "",
        "| Metric | Value [95% CI] |",
        "|---|---|",
        f"| Execution accuracy (result items) | {_fmt(s.get('exec_accuracy'))} |",
        f"| Execution accuracy, human-verified items only | {_fmt(s.get('exec_accuracy_verified_only'))} |",
        f"| Pass rate (all items) | {_fmt(s.get('pass_rate'))} |",
        f"| Unsafe accesses | {s.get('unsafe_accesses')} |",
        f"| Verified items in slice | {s.get('verified_items')} |",
    ]
    for k, v in (s.get("behaviour") or {}).items():
        L.append(f"| {k.replace('_', ' ')} | {_fmt(v)} |")
    for k, v in (s.get("pass_k") or {}).items():
        L.append(f"| {k} | {v:.3f} |")
    for title, key in (
        ("By set", "by_set"),
        ("By difficulty (golden)", "by_difficulty"),
        ("By trap (golden)", "by_trap"),
        ("By language", "by_language"),
    ):
        if s.get(key):
            L += ["", f"## {title}", "", "| Group | Pass rate [95% CI] |", "|---|---|"]
            L += [f"| {k} | {_fmt(v)} |" for k, v in s[key].items()]
    if s.get("failure_modes"):
        L += [
            "",
            "## Failure modes (code checks; F1-F3 attribution is a heuristic)",
            "",
            "| Mode | Name | Failure rate [95% CI] |",
            "|---|---|---|",
        ]
        L += [f"| {m} | {v['name']} | {_fmt(v)} |" for m, v in s["failure_modes"].items()]
    if s.get("judges"):
        L += [
            "",
            "## Judges",
            "",
            "| Judge | Observed FAIL | Corrected prevalence | TPR / TNR |",
            "|---|---|---|---|",
        ]
        for j, v in s["judges"].items():
            a = v.get("alignment") or {}
            cp = v.get("corrected_fail_prevalence")
            L.append(
                f"| {j} | {_fmt(v['observed_fail'])} | {(f'{cp:.3f}') if cp is not None else '---'} | "
                f"{('{:.2f} / {:.2f}'.format(a['tpr'], a['tnr'])) if a else '---'} |"
            )
    c = s.get("components") or {}
    if c:
        L += [
            "",
            "## Components",
            "",
            f"- Schema recall: {c.get('schema_recall')}",
            f"- Table-set match: {c.get('table_set_match')}",
            f"- Latency p50/p95 (all): {c.get('latency_all')}",
            f"- Mean tool steps {c.get('tool_steps_mean')}, LLM calls {c.get('llm_calls_mean')}, "
            f"input tokens {c.get('tokens_in_mean')}, output tokens {c.get('tokens_out_mean')}",
            f"- Shadow cost (paid-tier list prices, not spent): ${c.get('shadow_cost_usd_total', 0):.4f}",
        ]
    L += ["", "## Models that answered", ""] + [
        f"- {m}: {n} responses" for m, n in (s.get("models_answered") or {}).items()
    ]
    L.append(f"- Fallback answers: {s.get('fallback_answers')}")
    return "\n".join(L) + "\n"


def write_report(
    run_dir: Path, stopped: str | None = None, alignment: dict[str, dict[str, float]] | None = None
) -> dict[str, Any]:
    s = summarize(run_dir, alignment)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(s, stopped), encoding="utf-8")
    return s
