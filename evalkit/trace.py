"""Tracing: traces, spans and annotations in SQLite (optional Langfuse exporter).

Every LLM call, tool call, guard decision and SQL execution is a span. A trace ties a run to the exact
system version (model, prompt hash, config hash, git SHA). Annotations hold *human* labels only (rule 5).
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("span", default=None)
_current_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace", default=None)
_current_store: contextvars.ContextVar[TraceStore | None] = contextvars.ContextVar("store", default=None)

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY, scenario_id TEXT, run_id TEXT, role TEXT, question TEXT, config TEXT, model TEXT,
  prompt_hash TEXT, config_hash TEXT, git_sha TEXT, snapshot TEXT, started REAL, ended REAL,
  latency_s REAL, tokens_in INTEGER, tokens_out INTEGER, shadow_cost_usd REAL, n_llm_calls INTEGER,
  n_tool_calls INTEGER, status TEXT, final_text TEXT, final_sql TEXT, final_rows TEXT,
  models_used TEXT, fallback_used INTEGER, attrs TEXT
);
CREATE TABLE IF NOT EXISTS spans (
  id TEXT PRIMARY KEY, trace_id TEXT, parent_id TEXT, kind TEXT, name TEXT, start REAL, "end" REAL,
  duration_s REAL, error TEXT, attrs TEXT
);
CREATE INDEX IF NOT EXISTS spans_trace ON spans (trace_id, start);
CREATE TABLE IF NOT EXISTS annotations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, labeller TEXT, task TEXT, verdict TEXT,
  critique TEXT, failure_modes TEXT, split TEXT, created REAL
);
CREATE INDEX IF NOT EXISTS ann_trace ON annotations (trace_id);
"""


def git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            or "nogit"
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"


class TraceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def write_span(self, rec: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                'INSERT OR REPLACE INTO spans (id, trace_id, parent_id, kind, name, start, "end", duration_s, '
                "error, attrs) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    rec["id"],
                    rec["trace_id"],
                    rec["parent_id"],
                    rec["kind"],
                    rec["name"],
                    rec["start"],
                    rec["end"],
                    rec["end"] - rec["start"],
                    rec["error"],
                    json.dumps(rec["attrs"], default=str, ensure_ascii=False),
                ),
            )

    def write_trace(self, rec: dict[str, Any]) -> None:
        cols = [
            "id",
            "scenario_id",
            "run_id",
            "role",
            "question",
            "config",
            "model",
            "prompt_hash",
            "config_hash",
            "git_sha",
            "snapshot",
            "started",
            "ended",
            "latency_s",
            "tokens_in",
            "tokens_out",
            "shadow_cost_usd",
            "n_llm_calls",
            "n_tool_calls",
            "status",
            "final_text",
            "final_sql",
            "final_rows",
            "models_used",
            "fallback_used",
            "attrs",
        ]
        vals = []
        for k in cols:
            v = rec.get(k)
            if k in ("final_rows", "models_used", "attrs"):
                v = json.dumps(v, default=str, ensure_ascii=False)
            vals.append(v)
        with self._lock, self._conn() as c:
            c.execute(
                f"INSERT OR REPLACE INTO traces ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                vals,
            )

    # ---- reading
    def trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM traces WHERE id=?", (trace_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        for k in ("final_rows", "models_used", "attrs"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        return d

    def spans(self, trace_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM spans WHERE trace_id=? ORDER BY start", (trace_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["attrs"] = json.loads(d["attrs"]) if d.get("attrs") else {}
            out.append(d)
        return out

    def list_traces(self, run_id: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        q = "SELECT id, scenario_id, run_id, question, status, latency_s, n_tool_calls, model FROM traces"
        args: tuple[Any, ...] = ()
        if run_id:
            q += " WHERE run_id=?"
            args = (run_id,)
        q += " ORDER BY started DESC LIMIT ?"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, (*args, limit)).fetchall()]

    # ---- human annotations (written only from the viewer by a person)
    def add_annotation(
        self,
        trace_id: str,
        labeller: str,
        task: str,
        verdict: str,
        critique: str,
        failure_modes: list[str],
        split: str | None = None,
    ) -> None:
        if not labeller or labeller.lower() in {"claude", "llm", "model", "auto"}:
            raise ValueError("annotations must come from a named human labeller (rule 5)")
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO annotations (trace_id, labeller, task, verdict, critique, failure_modes, split, "
                "created) VALUES (?,?,?,?,?,?,?,?)",
                (trace_id, labeller, task, verdict, critique, json.dumps(failure_modes), split, time.time()),
            )

    def annotations(self, task: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM annotations"
        args: tuple[Any, ...] = ()
        if task:
            q += " WHERE task=?"
            args = (task,)
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q + " ORDER BY created", args).fetchall()]
        for r in rows:
            r["failure_modes"] = json.loads(r["failure_modes"] or "[]")
        return rows


_STORE: TraceStore | None = None


def default_store() -> TraceStore:
    global _STORE
    if _STORE is None:
        root = Path(__file__).resolve().parents[1]
        _STORE = TraceStore(os.environ.get("QUERYLENS_TRACE_DB", root / "data" / "traces.sqlite"))
    return _STORE


def set_store(store: TraceStore | None) -> None:
    global _STORE
    _STORE = store


class Span:
    """Context manager: ``with Span("tool", "run_sql", query=sql) as s: ...; s.set(rows=n)``."""

    def __init__(self, kind: str, name: str, store: TraceStore | None = None, **attrs: Any) -> None:
        self.store = store or _current_store.get() or default_store()
        self.rec: dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "kind": kind,
            "name": name,
            "attrs": dict(attrs),
            "trace_id": _current_trace.get(),
            "parent_id": None,
            "error": None,
        }
        self._tok: contextvars.Token[Span | None] | None = None

    @property
    def id(self) -> str:
        return self.rec["id"]

    def set(self, **kv: Any) -> None:
        self.rec["attrs"].update(kv)

    def __enter__(self) -> Span:
        parent = _current_span.get()
        self.rec["parent_id"] = parent.id if parent else None
        self.rec["start"] = time.time()
        self._tok = _current_span.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        self.rec["end"] = time.time()
        self.rec["error"] = repr(exc) if exc else None
        if self._tok is not None:
            _current_span.reset(self._tok)
        if self.rec["trace_id"]:
            try:
                self.store.write_span(self.rec)
            except sqlite3.Error as e:  # tracing must never break the agent
                log.error("trace write failed: %s", e)


class Trace:
    """Context manager that opens a trace; spans created inside attach to it."""

    def __init__(self, store: TraceStore | None = None, **fields: Any) -> None:
        self.store = store or default_store()
        self.rec: dict[str, Any] = {"id": uuid.uuid4().hex, "git_sha": git_sha(), **fields}
        self._tok: contextvars.Token[str | None] | None = None

    @property
    def id(self) -> str:
        return self.rec["id"]

    def set(self, **kv: Any) -> None:
        self.rec.update(kv)

    def __enter__(self) -> Trace:
        self.rec["started"] = time.time()
        self._tok = _current_trace.set(self.id)
        self._stok = _current_store.set(self.store)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        self.rec["ended"] = time.time()
        self.rec["latency_s"] = round(self.rec["ended"] - self.rec["started"], 3)
        if exc is not None:
            self.rec["status"] = "error"
            self.rec.setdefault("attrs", {})["error"] = repr(exc)
        if self._tok is not None:
            _current_trace.reset(self._tok)
            _current_store.reset(self._stok)
        self.store.write_trace(self.rec)


class LangfuseExporter:
    """Optional: push a stored trace to Langfuse if LANGFUSE_* keys are set and the SDK is installed."""

    def __init__(self) -> None:
        self.enabled = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))

    def export(self, store: TraceStore, trace_id: str) -> bool:
        if not self.enabled:
            return False
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found]
        except ImportError:
            log.warning("langfuse SDK not installed; skipping export")
            return False
        lf = Langfuse()
        t = store.trace(trace_id) or {}
        lf_trace = lf.trace(id=trace_id, name=t.get("question", "")[:80], metadata=t)  # type: ignore[attr-defined]
        for s in store.spans(trace_id):
            lf_trace.span(id=s["id"], name=f"{s['kind']}:{s['name']}", metadata=s["attrs"])
        lf.flush()
        return True
