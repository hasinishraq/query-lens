"""``evalkit`` CLI: datasets, run, gate, report, align, frontier."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LABELS_DB = ROOT / "datasets" / "labels" / "annotations.sqlite"


def cmd_datasets(a: argparse.Namespace) -> int:
    from .datasets import build_gold_results, report

    if a.action == "build":
        from .scenarios.build import build

        sets = build(ROOT / "data" / "bazaar_full.duckdb", ROOT / "data" / "bazaar_small.duckdb")
        print(json.dumps({k: len(v) for k, v in sets.items()}))
        return 0
    if a.action == "gold":
        bad = 0
        for snap in a.snapshots.split(","):
            r = build_gold_results(snap, ROOT / "data")
            print(f"{snap}: {r['n']} gold results, {len(r['problems'])} problems")
            for p in r["problems"]:
                print("  ", p)
            bad += len(r["problems"])
        return 1 if bad else 0
    rep, errors = report()
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    for e in errors:
        print("ERROR", e)
    print(f"datasets report: {'OK' if not errors else f'{len(errors)} rule violations'}")
    return 1 if errors else 0


def cmd_run(a: argparse.Namespace) -> int:
    from .runner import run

    if a.slice == "test" and not a.frozen:
        print(
            "Refusing to run the test split without --frozen (CLAUDE.md rule 3). Log it in docs/test_runs.md."
        )
        return 2
    run_dir = run(
        a.slice, a.size, a.repeats, a.config, run_id=a.run_id, resume=a.resume, judges=a.judges, limit=a.limit
    )
    print(f"run written to {run_dir}")
    print((run_dir / "report.md").read_text(encoding="utf-8")[:3000])
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    from .gate import write_gate

    g = write_gate(
        RESULTS / a.run,
        Path(a.markdown) if a.markdown else None,
        RESULTS / a.baseline if a.baseline else None,
    )
    print(g.markdown())
    return 0 if g.passed else 1


def cmd_report(a: argparse.Namespace) -> int:
    from .report import write_report

    align = json.loads(Path(a.alignment).read_text()) if a.alignment else None
    write_report(RESULTS / a.run, alignment=align)
    md = (RESULTS / a.run / "report.md").read_text(encoding="utf-8")
    print(md)
    html = (
        "<html><meta charset='utf-8'><body><pre>"
        + md.replace("&", "&amp;").replace("<", "&lt;")
        + "</pre></body></html>"
    )
    (RESULTS / a.run / "report.html").write_text(html, encoding="utf-8")
    return 0


def cmd_align(a: argparse.Namespace) -> int:
    """Judge validation: human labels (viewer) vs judge verdicts stored in a run."""
    from .align import align
    from .report import load_items
    from .trace import TraceStore

    store = TraceStore(LABELS_DB)
    labels = {
        r["trace_id"]: r
        for r in store.annotations(task=f"judge:{a.judge}")
        if not a.split or r["split"] == a.split
    }
    human, judge = [], []
    for it in load_items(RESULTS / a.run):
        v = (it.get("judges") or {}).get(a.judge)
        if v and it["trace_id"] in labels:
            human.append(labels[it["trace_id"]]["verdict"])
            judge.append(v["verdict"])
    if not human:
        print("no overlapping human labels and judge verdicts")
        return 1
    res = align(human, judge)
    print(json.dumps(res.as_dict(), indent=2))
    out = RESULTS / a.run / "alignment.json"
    cur = json.loads(out.read_text()) if out.exists() else {}
    cur[a.judge] = res.as_dict()
    out.write_text(json.dumps(cur, indent=2))
    return 0


def cmd_scenarios(a: argparse.Namespace) -> int:
    from .runner import run_scenarios

    if a.build:
        from .scenarios.pool import main as build_pool

        build_pool()
        return 0
    d = run_scenarios(a.config, a.size, limit=a.limit, run_id=a.run_id)
    print(f"scenario traces in {d} -- label them in the viewer (make viewer)")
    return 0


def cmd_frontier(a: argparse.Namespace) -> int:
    from .frontier import plot_frontier

    p = plot_frontier(a.slice, a.size, run_configs=a.configs.split(",") if a.configs else None, limit=a.limit)
    print(f"frontier written to {p}")
    return 0


def main(argv: list[str] | None = None) -> None:
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="evalkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("datasets")
    s.add_argument("action", choices=["report", "build", "gold"], nargs="?", default="report")
    s.add_argument("--snapshots", default="small,full")
    s.set_defaults(fn=cmd_datasets)
    s = sub.add_parser("run")
    s.add_argument("--slice", default="dev")
    s.add_argument("--size", default="full")
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--config", default="b_mistral")
    s.add_argument("--run-id", default=None)
    s.add_argument("--resume", default=None)
    s.add_argument("--judges", action="store_true")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument(
        "--frozen", action="store_true", help="required for the test split; human-approved runs only"
    )
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("gate")
    s.add_argument("--run", required=True)
    s.add_argument("--baseline", default=None)
    s.add_argument("--markdown", default=None)
    s.set_defaults(fn=cmd_gate)
    s = sub.add_parser("report")
    s.add_argument("--run", required=True)
    s.add_argument("--alignment", default=None)
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("align")
    s.add_argument("--run", required=True)
    s.add_argument("--judge", required=True, choices=["j1", "j2", "j3"])
    s.add_argument("--split", default="dev")
    s.set_defaults(fn=cmd_align)
    s = sub.add_parser("scenarios", help="build the 1,200-scenario pool or run the agent on it")
    s.add_argument("--build", action="store_true")
    s.add_argument("--config", default="b_mistral")
    s.add_argument("--size", default="full")
    s.add_argument("--limit", type=int, default=300)
    s.add_argument("--run-id", default=None)
    s.set_defaults(fn=cmd_scenarios)
    s = sub.add_parser("frontier")
    s.add_argument("--slice", default="dev")
    s.add_argument("--size", default="full")
    s.add_argument("--configs", default=None, help="comma list: also run these configs first")
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_frontier)
    a = ap.parse_args(argv)
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
