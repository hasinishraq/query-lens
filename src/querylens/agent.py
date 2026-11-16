"""Agent loop with budgets and stop conditions (CLAUDE.md 7.3)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evalkit.trace import Span, Trace, TraceStore

from .config import AgentConfig, ModelRef, load_runtime, shadow_cost_usd
from .llm import AllModelsFailed, LLMClient, LLMResponse, estimate_tokens
from .paths import PROMPTS, db_path
from .tools import TOOL_SPECS, Toolbox

log = logging.getLogger(__name__)


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]
    error: str | None = None
    guard_rule: str | None = None
    repeated: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    trace_id: str
    question: str
    role: str
    status: str  # answered | clarified | budget_exhausted | error
    stop_reason: str = ""
    final_text: str = ""
    final_sql: str = ""
    final_rows: list[Any] = Field(default_factory=list)
    clarifying_question: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    n_llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    models_used: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    escalated: bool = False
    shadow_cost_usd: float = 0.0
    prompt_hash: str = ""

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_calls)

    def sql_runs(self) -> list[ToolCallRecord]:
        return [t for t in self.tool_calls if t.name == "run_sql"]

    def first_search_tables(self) -> list[str]:
        for t in self.tool_calls:
            if t.name == "search_schema":
                return list(t.meta.get("retrieved_tables", []))
        return []

    def guard_rejections(self) -> list[str]:
        return [t.guard_rule for t in self.tool_calls if t.guard_rule]

    def last_result(self) -> tuple[list[str], list[list[Any]]] | None:
        for t in reversed(self.tool_calls):
            if t.name == "run_sql" and not t.error and "result_rows" in t.meta:
                return t.meta.get("result_columns", []), t.meta["result_rows"]
        return None


def load_prompt(name: str, role: str) -> tuple[str, str]:
    text = (PROMPTS / name).read_text(encoding="utf-8")
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return text.replace("{role}", role), h


def config_hash(cfg: AgentConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode("utf-8")).hexdigest()[:12]


class Agent:
    def __init__(
        self,
        cfg: AgentConfig,
        role: str = "analyst",
        size: str = "full",
        llm: LLMClient | None = None,
        store: TraceStore | None = None,
        db: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.role = role
        self.size = size
        self.llm = llm or LLMClient()
        self.store = store
        self.rt = load_runtime()
        self.db = str(db or db_path(size, role))
        self.system, self.prompt_hash = load_prompt(cfg.prompt, role)
        self.toolbox = Toolbox(role, self.db, self.rt)

    def close(self) -> None:
        self.toolbox.close()

    def _chain(self, escalated: bool) -> list[ModelRef]:
        if escalated and self.cfg.cascade:
            return [
                ModelRef(provider=self.cfg.cascade.provider, model=self.cfg.cascade.model),
                *self.cfg.fallbacks,
            ]
        return [self.cfg.agent, *self.cfg.fallbacks]

    def run(self, question: str, scenario_id: str | None = None, run_id: str | None = None) -> AgentResult:
        b = self.cfg.budgets
        t0 = time.perf_counter()
        trace = Trace(
            store=self.store,
            scenario_id=scenario_id,
            run_id=run_id,
            role=self.role,
            question=question,
            config=self.cfg.name,
            model=self.cfg.agent.model,
            prompt_hash=self.prompt_hash,
            config_hash=config_hash(self.cfg),
            snapshot=self.size,
        )
        res = AgentResult(
            trace_id=trace.id, question=question, role=self.role, status="error", prompt_hash=self.prompt_hash
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": question},
        ]
        seen_calls: dict[str, int] = {}
        escalated = False
        with trace:
            try:
                while True:
                    if res.n_llm_calls >= b.max_llm_calls:
                        res.status, res.stop_reason = "budget_exhausted", "max_llm_calls"
                        break
                    if time.perf_counter() - t0 > b.max_seconds:
                        res.status, res.stop_reason = "budget_exhausted", "max_seconds"
                        break
                    est_in = estimate_tokens(messages) + estimate_tokens(TOOL_SPECS)
                    if res.tokens_in + est_in > b.max_input_tokens and res.n_llm_calls > 0:
                        res.status, res.stop_reason = "budget_exhausted", "max_input_tokens"
                        break
                    left = b.max_llm_calls - res.n_llm_calls
                    if left == 2 and res.n_llm_calls > 0:
                        messages.append(
                            {
                                "role": "user",
                                "content": "[harness] 2 turns left: run your SQL now, then call "
                                "final_answer (or ask_user).",
                            }
                        )
                    elif left == 1 and res.n_llm_calls > 0:
                        messages.append(
                            {
                                "role": "user",
                                "content": "[harness] Last turn: call final_answer now with what "
                                "you have, or ask_user.",
                            }
                        )
                    self._compress_history(messages)
                    chain = self._chain(escalated)
                    with Span(
                        "llm",
                        "complete",
                        store=self.store,
                        chain=[c.model_dump() for c in chain],
                        n_messages=len(messages),
                    ) as span:
                        resp = self.llm.complete(messages, TOOL_SPECS, chain, self.cfg.params)
                        span.set(
                            provider=resp.provider,
                            requested_model=resp.requested_model,
                            returned_model=resp.returned_model,
                            tokens_in=resp.tokens_in,
                            tokens_out=resp.tokens_out,
                            latency_s=resp.latency_s,
                            cached=resp.cached,
                            fallback_used=resp.fallback_used,
                            content=resp.content,
                            tool_calls=[tc.model_dump() for tc in resp.tool_calls],
                            input_messages=messages[-3:],
                        )
                    self._account(res, resp)
                    if not resp.tool_calls:
                        text = (resp.content or "").strip()
                        if not text and res.n_llm_calls < b.max_llm_calls:
                            messages.append({"role": "assistant", "content": ""})
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "[harness] Call final_answer with the numbers "
                                    "from your last result.",
                                }
                            )
                            continue
                        # no tool call: the text is the answer; keep the last successful query as the final SQL
                        res.final_text = text
                        for t in reversed(res.tool_calls):
                            if t.name == "run_sql" and not t.error and t.meta.get("sql"):
                                res.final_sql = t.meta["sql"]
                                res.final_rows = t.meta.get("result_rows", [])[:200]
                                break
                        res.status, res.stop_reason = "answered", "text_reply"
                        break
                    messages.append(self._assistant_msg(resp))
                    done = False
                    for tc in resp.tool_calls:
                        if len(res.tool_calls) >= b.max_tool_calls:
                            res.status, res.stop_reason = "budget_exhausted", "max_tool_calls"
                            done = True
                            break
                        key = tc.name + json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)
                        seen_calls[key] = seen_calls.get(key, 0) + 1
                        if seen_calls[key] >= self.rt.agent.repeat_call_limit and tc.name not in (
                            "final_answer",
                            "ask_user",
                        ):
                            out = json.dumps(
                                {
                                    "error": f"You already called {tc.name} with these exact arguments. "
                                    "Use the earlier result or change the arguments."
                                }
                            )
                            res.tool_calls.append(
                                ToolCallRecord(
                                    name=tc.name, args=tc.arguments, repeated=True, error="repeated_call"
                                )
                            )
                            messages.append(self._tool_msg(tc.id, tc.name, out, resp))
                            continue
                        tr = self.toolbox.call(tc.name, tc.arguments)
                        rec = ToolCallRecord(
                            name=tc.name,
                            args=tc.arguments,
                            error=tr.error,
                            guard_rule=(tr.guard.rule if tr.guard and not tr.guard.ok else None),
                            meta=tr.meta,
                        )
                        res.tool_calls.append(rec)
                        if tr.terminal:
                            if tc.name == "final_answer":
                                res.final_text = tr.output.get("text", "")
                                res.final_sql = tr.output.get("sql", "")
                                res.final_rows = tr.output.get("rows", [])
                                res.status, res.stop_reason = "answered", "final_answer"
                            else:
                                res.clarifying_question = tr.output.get("question", "")
                                res.status, res.stop_reason = "clarified", "ask_user"
                            done = True
                            break
                        if self.cfg.cascade and not escalated and self._should_escalate(tr):
                            escalated = True
                            res.escalated = True
                        messages.append(
                            self._tool_msg(tc.id, tc.name, tr.as_message(b.tool_output_tokens), resp)
                        )
                    if done:
                        break
            except AllModelsFailed as e:
                res.status, res.stop_reason = "error", f"llm: {e}"[:500]
            finally:
                res.latency_s = round(time.perf_counter() - t0, 3)
                trace.set(
                    status=res.status,
                    final_text=res.final_text,
                    final_sql=res.final_sql,
                    final_rows=res.final_rows[:50],
                    tokens_in=res.tokens_in,
                    tokens_out=res.tokens_out,
                    n_llm_calls=res.n_llm_calls,
                    n_tool_calls=len(res.tool_calls),
                    shadow_cost_usd=round(res.shadow_cost_usd, 6),
                    models_used=res.models_used,
                    fallback_used=int(res.fallback_used),
                    attrs={
                        "stop_reason": res.stop_reason,
                        "clarifying_question": res.clarifying_question,
                        "escalated": res.escalated,
                        "guard_rejections": res.guard_rejections(),
                    },
                )
        return res

    @staticmethod
    def _compress_history(messages: list[dict[str, Any]], keep_last: int = 2, max_chars: int = 600) -> None:
        """Token diet: tool outputs older than the last ``keep_last`` are cut to ``max_chars`` characters."""
        idx = [
            i
            for i, m in enumerate(messages)
            if m.get("role") == "tool" or str(m.get("content", "")).startswith("Result of ")
        ]
        for i in idx[:-keep_last] if keep_last else idx:
            c = str(messages[i].get("content", ""))
            if len(c) > max_chars:
                messages[i] = {**messages[i], "content": c[:max_chars] + " ...[older output trimmed]"}

    def _should_escalate(self, tr: Any) -> bool:
        on = set(self.cfg.cascade.escalate_on) if self.cfg.cascade else set()
        if tr.name != "run_sql":
            return False
        if "guard_reject" in on and tr.guard is not None and not tr.guard.ok:
            return True
        if "sql_error" in on and tr.error and not (tr.guard is not None and not tr.guard.ok):
            return True
        return "empty_result" in on and not tr.error and tr.output.get("row_count", 1) == 0

    def _account(self, res: AgentResult, resp: LLMResponse) -> None:
        res.n_llm_calls += 1
        res.tokens_in += resp.tokens_in
        res.tokens_out += resp.tokens_out
        res.shadow_cost_usd += shadow_cost_usd(resp.requested_model, resp.tokens_in, resp.tokens_out)
        tag = f"{resp.provider}/{resp.returned_model}"
        if tag not in res.models_used:
            res.models_used.append(tag)
        res.fallback_used = res.fallback_used or resp.fallback_used

    @staticmethod
    def _assistant_msg(resp: LLMResponse) -> dict[str, Any]:
        if resp.tool_calls and resp.tool_calls[0].id.startswith("text-"):
            return {"role": "assistant", "content": resp.content or ""}
        return {
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.raw_arguments or json.dumps(tc.arguments)},
                    **(tc.extra or {}),
                }
                for tc in resp.tool_calls
            ],
        }

    @staticmethod
    def _tool_msg(call_id: str, name: str, content: str, resp: LLMResponse) -> dict[str, Any]:
        if call_id.startswith("text-"):
            return {"role": "user", "content": f"Result of {name}: {content}"}
        return {"role": "tool", "tool_call_id": call_id, "content": content}
