"""Datasets composition rules, gold sync, and an end-to-end runner check with a gold-oracle LLM."""

from __future__ import annotations

import json

import pytest

from evalkit.datasets import derive_gold_tables, gold_sql, load_all, report, select_slice
from evalkit.trace import TraceStore
from querylens.llm import LLMResponse, ToolCall, estimate_tokens


def test_datasets_report_passes():
    rep, errors = report()
    assert errors == [], errors[:10]
    assert rep["total"] == 930


def test_gold_tables_in_sync():
    for it in load_all():
        sql = gold_sql(it)
        if sql:
            assert derive_gold_tables(sql) == sorted(it.gold_tables), it.id


def test_no_human_labels_were_generated():
    """Rule 5: verified_by stays empty until the human verifies in the viewer."""
    assert all(
        i.verified_by is None or i.verified_by.lower() not in {"claude", "llm", "auto"} for i in load_all()
    )


def test_slices():
    assert len(select_slice("dev")) == 342
    assert len(select_slice("test")) == 426
    assert len(select_slice("train")) == 162
    assert len(select_slice("ci")) == 60


class GoldOracle:
    """Answers each golden question with its gold SQL: the pipeline must then score 100% execution accuracy."""

    def __init__(self, by_question: dict[str, str]) -> None:
        self.by_q = by_question

    def complete(self, messages, tools, chain, params=None):
        q = messages[1]["content"]
        sql = self.by_q[q]
        last = messages[-1]
        if last.get("role") == "tool":
            body = json.loads(last["content"])
            rows = body.get("rows", [])
            call = ToolCall(
                id="f",
                name="final_answer",
                arguments={"text": f"Result: {rows[:3]}", "sql": sql, "rows": rows},
            )
        else:
            call = ToolCall(id="r", name="run_sql", arguments={"query": sql})
        return LLMResponse(
            tool_calls=[call],
            provider="fake",
            requested_model="oracle",
            returned_model="oracle",
            tokens_in=estimate_tokens(messages),
            tokens_out=20,
        )


@pytest.mark.parametrize("n_items", [12])
def test_runner_end_to_end_with_oracle(tmp_path, monkeypatch, small_db, n_items):
    import evalkit.runner as R

    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    items = [i for i in select_slice("golden:train")][:n_items]
    oracle = GoldOracle({i.question: gold_sql(i) for i in items})
    monkeypatch.setattr(R, "select_slice", lambda s: items)
    run_dir = R.run(
        "golden:train",
        "small",
        1,
        "b_mistral",
        run_id="oracle-test",
        llm=oracle,  # type: ignore[arg-type]
        store=TraceStore(tmp_path / "t.sqlite"),
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_results"] == n_items
    assert summary["exec_accuracy"]["rate"] == 1.0
    assert summary["unsafe_accesses"] == 0
    # resume is a no-op when everything is done
    R.run(
        "golden:train",
        "small",
        1,
        "b_mistral",
        resume="oracle-test",
        llm=oracle,  # type: ignore[arg-type]
        store=TraceStore(tmp_path / "t.sqlite"),
    )
    lines = (run_dir / "items.jsonl").read_text().splitlines()
    assert len(lines) == n_items
