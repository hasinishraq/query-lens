"""Run an eval slice with repeats on a snapshot (CLAUDE.md 8.9, 7.4 runner requirements).

Each finished (item, repeat) is appended to ``results/<run_id>/items.jsonl`` immediately, so a quota stop
loses nothing; ``--resume <run_id>`` continues where it stopped, possibly days later.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from querylens.agent import Agent, AgentResult
from querylens.config import AgentConfig, load_config, load_providers
from querylens.guard import check_sql, open_role_connection, role_schema
from querylens.llm import AllModelsFailed, LLMClient
from querylens.paths import DATA, db_path
from querylens.ratelimit import QuotaExhausted

from . import checks as C
from .datasets import DATASETS, dataset_hash, gold_sql, load_gold_hashes, select_slice
from .fingerprint import fp_match, multi_fingerprint, result_match
from .judges import APPLIES_TO, Judge
from .schemas import EvalItem, ItemResult, RunInfo, Verdict
from .trace import TraceStore, git_sha

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def make_run_id(config: str, slice_name: str, snapshot: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M')}-{config}-{slice_name.replace(':', '_')}-{snapshot}"


def agent_view(res: AgentResult) -> C.AgentView:
    return C.AgentView(
        status=res.status,
        final_text=res.final_text,
        final_sql=res.final_sql,
        final_rows=res.final_rows,
        tool_calls=[t.model_dump() for t in res.tool_calls],
        clarifying_question=res.clarifying_question,
    )


class Evaluator:
    """Scores one agent run against one item (code checks; judges optional)."""

    def __init__(self, snapshot: str, role: str = "analyst", judges: dict[str, Judge] | None = None) -> None:
        self.snapshot = snapshot
        self.db = db_path(snapshot, role)
        self.schema = role_schema(str(self.db), role)
        from querylens.config import SqlRuntime

        self.rt = SqlRuntime()
        self.con = open_role_connection(self.db, self.rt)
        self.gold = load_gold_hashes(snapshot)
        self.judges = judges or {}

    def close(self) -> None:
        self.con.close()

    def _gold_rows(self, item: EvalItem) -> list[list[Any]]:
        key = Path(item.gold_sql_file or "").stem
        p = DATASETS / "gold_results" / self.snapshot / f"{key}.parquet"
        if not p.exists():
            return []
        rows = duckdb.sql(f"SELECT * FROM read_parquet('{p.as_posix()}')").fetchall()
        return [list(r) for r in rows]

    def _pred(self, sql: str) -> tuple[dict[str, list[Any]] | None, list[list[Any]]]:
        if not sql or not check_sql(sql, self.schema).ok:
            return None, []
        try:
            fp = multi_fingerprint(self.con, sql)
            rows = [list(r) for r in self.con.execute(f"SELECT * FROM ({sql}) _p LIMIT 1000").fetchall()]
            return fp, rows
        except duckdb.Error:
            return None, []

    def score(
        self, item: EvalItem, res: AgentResult, run_id: str, repeat: int, difficulty_max_calls: int = 12
    ) -> ItemResult:
        v = agent_view(res)
        beh = C.behaviour(v)
        key = Path(item.gold_sql_file or "").stem
        g = self.gold.get(key, {})
        gold_fp = g.get("fp")
        trap_fp = g.get("trap_fp")
        exec_ok: bool | None = None
        pred_fp, pred_rows = (None, [])
        gold_rows: list[list[Any]] = []
        if item.expected.type == "result" and item.gold_sql_file:
            pred_fp, pred_rows = self._pred(res.final_sql)
            gold_rows = self._gold_rows(item)
            if pred_fp is not None:
                exec_ok = fp_match(pred_fp, gold_fp)
            else:
                exec_ok = bool(res.final_rows) and result_match(
                    res.final_rows, gold_rows, item.expected.float_tol, item.expected.ordered
                )
        checks: list[Any] = []
        f4 = C.check_f4_hallucination(v)
        if item.expected.type == "result" and item.gold_sql_file:
            gsql = gold_sql(item) or ""
            ok = bool(exec_ok)
            checks += [
                C.check_f1_fanout(res.final_sql, pred_rows, gold_rows, ok),
                C.check_f2_filter(res.final_sql, gsql, ok),
                C.check_f3_aggregation(res.final_sql, gsql, ok),
                f4,
                C.check_f6_numbers(v, pred_rows or res.final_rows, item.question),
                C.check_f9_schema_linking(res.final_sql, item.gold_tables, ok, f4),
                C.check_f10_metric(v, item.dims.metric, ok),
                C.check_f11_trap(_first(pred_fp), _first(trap_fp), item.dims.trap),
            ]
        else:
            checks.append(f4)
        f7 = C.check_f7_unsafe(v)
        checks += [f7, C.check_f8_waste(v, item.dims.difficulty, difficulty_max_calls)]
        verdicts: dict[str, Verdict] = {}
        for name, judge in self.judges.items():
            if item.set in APPLIES_TO[name]:
                verdicts[name] = judge.grade(judge_input(item, res, pred_rows))
        return ItemResult(
            run_id=run_id,
            item_id=item.id,
            repeat=repeat,
            split=item.split,
            set=item.set,
            dims=item.dims.model_dump(),
            expected_type=item.expected.type,
            trace_id=res.trace_id,
            status=res.status,
            behaviour=beh,
            exec_match=exec_ok,
            pred_fingerprint=_first(pred_fp),
            gold_fingerprint=_first(gold_fp),
            checks=checks,
            judges=verdicts,
            unsafe_access=not f7.passed,
            tool_calls=res.n_tool_calls,
            llm_calls=res.n_llm_calls,
            tokens_in=res.tokens_in,
            tokens_out=res.tokens_out,
            latency_s=res.latency_s,
            shadow_cost_usd=res.shadow_cost_usd,
            models_used=res.models_used,
            fallback_used=res.fallback_used,
            first_search_tables=res.first_search_tables(),
            final_tables=sorted(C.tables_in_sql(res.final_sql)),
            verified=bool(item.verified_by),
            finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )


def _first(fp: dict[str, list[Any]] | None) -> list[Any] | None:
    return list(fp["d"]) if fp and "d" in fp else None


def judge_input(item: EvalItem, res: AgentResult, rows: list[list[Any]]) -> dict[str, Any]:
    steps = []
    for t in res.tool_calls:
        s = f"{t.name}({json.dumps(t.args, ensure_ascii=False)[:300]})"
        if t.error:
            s += f" -> ERROR {t.error[:200]}"
        steps.append(s)
    return {
        "question": item.question,
        "role": item.role,
        "steps": steps,
        "final_sql": res.final_sql,
        "final_text": res.final_text,
        "clarifying_question": res.clarifying_question,
        "result_rows": rows[:30] if rows else res.final_rows[:30],
    }


def build_judges(cfg: AgentConfig, client: LLMClient) -> dict[str, Judge]:
    fam = load_providers()[cfg.agent.provider].family_of(cfg.agent.model)
    chain = [cfg.judge, *cfg.judge_fallbacks]
    return {n: Judge.load(n, chain, agent_family=fam, client=client) for n in ("j1", "j2", "j3")}


def run(
    slice_name: str,
    size: str,
    repeats: int,
    config: str,
    run_id: str | None = None,
    resume: str | None = None,
    judges: bool = False,
    limit: int | None = None,
    llm: LLMClient | None = None,
    store: TraceStore | None = None,
) -> Path:
    if resume:
        run_dir = RESULTS / resume
        info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
        slice_name, size, repeats, config, judges = (
            info.slice,
            info.snapshot,
            info.repeats,
            info.config,
            info.judges,
        )
        run_id = info.run_id
    cfg = load_config(config)
    items = select_slice(slice_name)
    if limit:
        items = items[:limit]
    run_id = run_id or make_run_id(cfg.name, slice_name, size)
    run_dir = RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    client = llm or LLMClient()
    store = store or TraceStore(DATA / "traces.sqlite")
    agents: dict[str, Agent] = {}
    if not resume:
        a0 = Agent(cfg, role="analyst", size=size, llm=client, store=store)
        agents["analyst"] = a0
        info = RunInfo(
            run_id=run_id,
            config=cfg.name,
            git_sha=git_sha(),
            snapshot=size,
            slice=slice_name,
            repeats=repeats,
            model=f"{cfg.agent.provider}/{cfg.agent.model}",
            prompt_hash=a0.prompt_hash,
            dataset_hash=dataset_hash(items),
            started=datetime.now(UTC).isoformat(timespec="seconds"),
            item_ids=[i.id for i in items],
            judges=judges,
        )
        (run_dir / "run.json").write_text(info.model_dump_json(indent=2), encoding="utf-8")
    done: set[tuple[str, int]] = set()
    items_path = run_dir / "items.jsonl"
    if items_path.exists():
        for line in items_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["item_id"], d["repeat"]))
    evaluator = Evaluator(size, judges=build_judges(cfg, client) if judges else None)
    stopped = None
    try:
        with open(items_path, "a", encoding="utf-8") as f:
            for rep in range(repeats):
                for it in items:
                    if (it.id, rep) in done:
                        continue
                    agent = agents.get(it.role) or agents.setdefault(
                        it.role, Agent(cfg, role=it.role, size=size, llm=client, store=store)
                    )
                    try:
                        res = agent.run(it.question, scenario_id=it.id, run_id=run_id)
                        if res.status == "error" and "quota" in res.stop_reason.lower():
                            raise QuotaExhausted("?", "?", res.stop_reason, "later")
                        ir = evaluator.score(it, res, run_id, rep)
                    except (QuotaExhausted, AllModelsFailed) as e:
                        stopped = str(e)
                        raise
                    f.write(ir.model_dump_json() + "\n")
                    f.flush()
    except (QuotaExhausted, AllModelsFailed):
        log.warning("run stopped (%s). Resume later with: evalkit run --resume %s", stopped, run_id)
    finally:
        evaluator.close()
        for a in agents.values():
            a.close()
    from .report import write_report

    write_report(run_dir, stopped=stopped)
    return run_dir


def run_scenarios(
    config: str,
    size: str = "full",
    limit: int | None = None,
    run_id: str | None = None,
    llm: LLMClient | None = None,
    store: TraceStore | None = None,
) -> Path:
    """Phase 5: run the agent on the scenario pool (questions only) to produce traces for human labelling.
    Writes ``results/<run_id>/scenarios.jsonl`` (scenario id -> trace id). Resumable: done scenarios are skipped."""
    cfg = load_config(config)
    pool = [
        json.loads(x)
        for x in (DATASETS / "scenarios.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if limit:
        pool = pool[:limit]
    run_id = run_id or make_run_id(cfg.name, "scenarios", size)
    run_dir = RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "scenarios.jsonl"
    done = (
        {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
        if out.exists()
        else set()
    )
    agent = Agent(
        cfg,
        role="analyst",
        size=size,
        llm=llm or LLMClient(),
        store=store or TraceStore(DATA / "traces.sqlite"),
    )
    try:
        with open(out, "a", encoding="utf-8") as f:
            for s in pool:
                if s["id"] in done:
                    continue
                try:
                    res = agent.run(s["question"], scenario_id=s["id"], run_id=run_id)
                except (QuotaExhausted, AllModelsFailed) as e:
                    log.warning("stopped (%s); rerun with --run-id %s to resume", e, run_id)
                    break
                f.write(
                    json.dumps(
                        {"id": s["id"], "trace_id": res.trace_id, "status": res.status, "dims": s["dims"]}
                    )
                    + "\n"
                )
                f.flush()
    finally:
        agent.close()
    return run_dir
