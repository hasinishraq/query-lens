"""Re-apply the current code checks to a finished run without calling any model.

Used when a check itself is fixed (e.g. a false-positive regex). Rebuilds each item's agent view from the
stored trace, recomputes F7/F8 and unsafe_access, keeps every other field, and rewrites report.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import checks as C
from .report import write_report
from .trace import TraceStore


def rescore_run(run_dir: Path, store: TraceStore) -> int:
    path = run_dir / "items.jsonl"
    items = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    changed = 0
    for it in items:
        t = store.trace(it["trace_id"]) or {}
        calls = [
            {
                "name": s["name"],
                "guard_rule": s["attrs"].get("guard_rule"),
                "repeated": False,
                "meta": {k: s["attrs"].get(k) for k in ("timed_out",)},
            }
            for s in store.spans(it["trace_id"])
            if s["kind"] == "tool"
        ]
        v = C.AgentView(
            status=t.get("status", ""),
            final_text=t.get("final_text") or "",
            final_rows=t.get("final_rows") or [],
            tool_calls=calls,
        )
        f7 = C.check_f7_unsafe(v)
        old = it["unsafe_access"]
        it["unsafe_access"] = not f7.passed
        it["checks"] = [f7.model_dump() if c["mode"] == "F7" else c for c in it["checks"]]
        changed += int(old != it["unsafe_access"])
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    write_report(run_dir)
    return changed
