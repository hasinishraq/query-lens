"""Tier 1: mocked integration of the agent loop with recorded (scripted) replies on the small size."""

from __future__ import annotations

import pytest

from evalkit.trace import TraceStore
from querylens.agent import Agent, load_prompt
from querylens.config import load_config, load_runtime
from querylens.llm import LLMResponse, ToolCall, estimate_tokens
from querylens.tools import TOOL_SPECS


class ScriptedLLM:
    """Returns pre-recorded responses in order (stand-in for the disk cache in replay mode)."""

    def __init__(self, script: list[list[tuple[str, dict]] | str]) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def complete(self, messages, tools, chain, params=None):
        self.calls.append(list(messages))
        step = self.script.pop(0)
        if isinstance(step, str):
            return LLMResponse(
                content=step,
                provider="fake",
                requested_model="m",
                returned_model="m-v1",
                tokens_in=estimate_tokens(messages),
                tokens_out=10,
            )
        calls = [ToolCall(id=f"c{len(self.calls)}_{i}", name=n, arguments=a) for i, (n, a) in enumerate(step)]
        return LLMResponse(
            tool_calls=calls,
            provider="fake",
            requested_model="m",
            returned_model="m-v1",
            tokens_in=estimate_tokens(messages),
            tokens_out=30,
        )


@pytest.fixture
def store(tmp_path):
    return TraceStore(tmp_path / "traces.sqlite")


def _agent(analyst_db, store, script):
    llm = ScriptedLLM(script)
    a = Agent(load_config("b_mistral"), role="analyst", size="small", llm=llm, store=store)  # type: ignore[arg-type]
    return a, llm


SQL = "SELECT COUNT(*) AS n FROM orders WHERE NOT is_deleted AND status NOT IN ('cancelled','canceled','payment_failed')"


def test_happy_path_and_trace(analyst_db, store):
    a, llm = _agent(
        analyst_db,
        store,
        [
            [("search_schema", {"query": "orders count"})],
            [("get_metric", {"name": "order_count"})],
            [("run_sql", {"query": SQL})],
            [
                (
                    "final_answer",
                    {
                        "text": "There were 4,863 valid orders (metric order_count).",
                        "sql": SQL,
                        "rows": [[4863]],
                    },
                )
            ],
        ],
    )
    try:
        r = a.run("How many orders did we have?", scenario_id="t-1", run_id="test")
    finally:
        a.close()
    assert r.status == "answered" and r.n_llm_calls == 4 and r.n_tool_calls == 4
    assert r.first_search_tables()[:3]
    assert "orders" in r.first_search_tables()
    assert r.last_result() is not None
    t = store.trace(r.trace_id)
    assert t["status"] == "answered" and t["prompt_hash"] == r.prompt_hash
    kinds = [s["kind"] for s in store.spans(r.trace_id)]
    for k in ("llm", "tool", "guard", "sql"):
        assert k in kinds
    llm_span = next(s for s in store.spans(r.trace_id) if s["kind"] == "llm")
    assert llm_span["attrs"]["returned_model"] == "m-v1"
    # tool output went back to the model as a tool message
    assert any(m.get("role") == "tool" for m in llm.calls[-1])


def test_guard_rejection_is_recorded(analyst_db, store):
    a, _ = _agent(
        analyst_db,
        store,
        [
            [("run_sql", {"query": "SELECT email FROM customers"})],
            [
                (
                    "final_answer",
                    {"text": "Email is not available to the analyst role.", "sql": "", "rows": []},
                )
            ],
        ],
    )
    try:
        r = a.run("List customer emails")
    finally:
        a.close()
    assert r.guard_rejections() == ["forbidden_column"]
    assert any(s["kind"] == "guard" and s["attrs"]["decision"] == "reject" for s in store.spans(r.trace_id))


def test_repeated_call_detected(analyst_db, store):
    q = {"query": "orders"}
    a, llm = _agent(analyst_db, store, [[("search_schema", q)], [("search_schema", q)], "done"])
    try:
        r = a.run("x")
    finally:
        a.close()
    assert r.tool_calls[1].repeated is True
    assert "already called" in llm.calls[-1][-1]["content"]


def test_ask_user_ends_turn(analyst_db, store):
    a, _ = _agent(analyst_db, store, [[("ask_user", {"question": "Which metric: GMV or net revenue?"})]])
    try:
        r = a.run("Who are our best sellers?")
    finally:
        a.close()
    assert r.status == "clarified" and "GMV" in r.clarifying_question


def test_llm_budget(analyst_db, store):
    script = [[("search_schema", {"query": f"q{i}"})] for i in range(10)]
    a, _ = _agent(analyst_db, store, script)
    try:
        r = a.run("x")
    finally:
        a.close()
    assert r.status == "budget_exhausted"
    assert r.stop_reason in ("max_llm_calls", "max_input_tokens")
    assert r.n_llm_calls <= load_config("b_mistral").budgets.max_llm_calls


def test_untrusted_text_wrapped(analyst_db, store):
    a, llm = _agent(
        analyst_db, store, [[("run_sql", {"query": "SELECT body FROM product_reviews LIMIT 3"})], "ok"]
    )
    try:
        a.run("show reviews")
    finally:
        a.close()
    tool_msg = next(m for m in llm.calls[-1] if m.get("role") == "tool")
    assert "<untrusted_data source=" in tool_msg["content"]


def test_output_pii_redacted(analyst_db, store):
    a, _ = _agent(
        analyst_db,
        store,
        [[("final_answer", {"text": "Contact rahim@gmail.com or +8801712345678", "sql": "", "rows": []})]],
    )
    try:
        r = a.run("x")
    finally:
        a.close()
    assert "REDACTED" in r.final_text and "rahim@gmail.com" not in r.final_text


def test_prompt_token_budget():
    """The system prompt and tool specs stay inside the token diet (CLAUDE.md 7.4)."""
    text, _ = load_prompt("system_v1.md", "analyst")
    assert estimate_tokens(text) <= load_runtime().prompt.max_system_tokens
    assert estimate_tokens(text) + estimate_tokens(TOOL_SPECS) <= 3000
