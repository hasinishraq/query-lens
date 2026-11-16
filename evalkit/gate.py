"""CI gate (CLAUDE.md 8.8): apply thresholds from configs/ci.yaml to a run and render a PR comment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .report import _passed, load_items, summarize

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GateResult:
    passed: bool
    rows: list[tuple[str, str, str, bool]]

    def markdown(self) -> str:
        lines = [
            "### QueryLens eval gate: " + ("PASS" if self.passed else "FAIL"),
            "",
            "| Check | Value | Threshold | Result |",
            "|---|---|---|---|",
        ]
        lines += [f"| {a} | {b} | {c} | {'pass' if ok else 'FAIL'} |" for a, b, c, ok in self.rows]
        return "\n".join(lines) + "\n"


def thresholds() -> dict[str, Any]:
    with open(ROOT / "configs" / "ci.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["gate"]


def gate(run_dir: Path, baseline_dir: Path | None = None) -> GateResult:
    t = thresholds()
    s = summarize(run_dir)
    rows: list[tuple[str, str, str, bool]] = []
    acc = (s.get("exec_accuracy") or {}).get("rate")
    rows.append(
        (
            "Execution accuracy",
            f"{acc:.3f}" if acc is not None else "---",
            f">= {t['min_exec_accuracy']}",
            acc is not None and acc >= t["min_exec_accuracy"],
        )
    )
    if baseline_dir is not None and (baseline_dir / "items.jsonl").exists():
        b = summarize(baseline_dir)
        bacc = (b.get("exec_accuracy") or {}).get("rate")
        if acc is not None and bacc is not None:
            drop = bacc - acc
            rows.append(
                ("Drop vs main", f"{drop:+.3f}", f"<= {t['max_drop_vs_main']}", drop <= t["max_drop_vs_main"])
            )
    ua = int(s.get("unsafe_accesses", 0))
    rows.append(
        ("Unsafe accesses", str(ua), f"<= {t['max_unsafe_accesses']}", ua <= t["max_unsafe_accesses"])
    )
    reg = [i for i in load_items(run_dir) if i["set"] == "regression" and i["repeat"] == 0]
    new_fail = sum(1 for i in reg if not _passed(i))
    rows.append(
        (
            "Regression-set failures",
            str(new_fail),
            f"<= {t['allow_new_regression_failures']}",
            new_fail <= t["allow_new_regression_failures"],
        )
    )
    complete = s.get("n_results", 0) >= s.get("n_planned", 0) > 0
    rows.append(("Run complete", f"{s.get('n_results')}/{s.get('n_planned')}", "all items", complete))
    return GateResult(passed=all(r[3] for r in rows), rows=rows)


def write_gate(run_dir: Path, out: Path | None, baseline: Path | None) -> GateResult:
    g = gate(run_dir, baseline)
    if out:
        out.write_text(g.markdown(), encoding="utf-8")
    (run_dir / "gate.json").write_text(
        json.dumps({"passed": g.passed, "rows": g.rows}, indent=2), encoding="utf-8"
    )
    return g
