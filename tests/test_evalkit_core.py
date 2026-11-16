"""Unit tests for fingerprint, stats, checks (pass + fail example each), align, judges parsing."""

import math

import pytest

from bazaar_world.mysqldb import admin_connect
from evalkit import checks as C
from evalkit.align import align
from evalkit.fingerprint import (
    fp_match,
    multi_fingerprint,
    result_match,
    rows_fingerprint,
    rows_multi_fingerprint,
    sql_fingerprint,
)
from evalkit.judges import Judge, parse_verdict
from evalkit.stats import (
    bootstrap_ci,
    cohens_kappa,
    mcnemar_exact,
    paired_comparison,
    pass_at_k,
    pass_hat_k,
    rogan_gladen,
    rogan_gladen_ci,
    standard_error,
    wilson_ci,
)
from querylens.config import ModelRef


# ------------------------------------------------------------------ fingerprint (MySQL)
@pytest.fixture
def con(mysql_up):
    c = admin_connect()
    yield c
    c.close()


def test_fingerprint_ignores_order_and_names(con):
    a = sql_fingerprint(con, "SELECT 1 AS x, 'A' AS y UNION ALL SELECT 2, 'b'")
    b = sql_fingerprint(con, "SELECT 'B' AS name, 2 AS n UNION ALL SELECT 'a', 1")
    assert a[:2] == b[:2]


def test_fingerprint_counts_duplicates_and_rounds(con):
    a = sql_fingerprint(con, "SELECT 1.00001 AS x UNION ALL SELECT 1.00001")
    b = sql_fingerprint(con, "SELECT 1.00002 AS x")
    c = sql_fingerprint(con, "SELECT 1.0 AS x UNION ALL SELECT 1.0")
    assert a[:2] != b[:2]
    assert a[:2] == c[:2]


def test_fingerprint_int_vs_double_and_date_vs_midnight(con):
    a = sql_fingerprint(con, "SELECT 5 AS n, DATE '2025-01-01' AS d")
    b = sql_fingerprint(con, "SELECT 5.0 AS n, TIMESTAMP '2025-01-01 00:00:00' AS d")
    assert a[:2] == b[:2]


def test_multi_fingerprint_absorbs_boundary_noise(con):
    a = multi_fingerprint(con, "SELECT 454731.49345000001 AS x")
    b = multi_fingerprint(con, "SELECT 454731.49344999999 AS x")
    c = multi_fingerprint(con, "SELECT 454731.4999 AS x")
    assert fp_match(a, b)
    assert not fp_match(a, c)


def test_rows_multi_fingerprint_matches_sql(con):
    rows = [[1, "a"], [2, "b"]]
    assert rows_multi_fingerprint(rows) == multi_fingerprint(con, "SELECT 2, 'B' UNION ALL SELECT 1, 'a'")


def test_rows_fingerprint_and_result_match():
    assert rows_fingerprint([[1, "a"], [2, "b"]]) == rows_fingerprint([["B", 2.0], ["a", 1]])
    assert result_match([[1, 2.00001]], [[2.0, 1]], float_tol=1e-3)
    assert not result_match([[1, 2.1]], [[2.0, 1]], float_tol=1e-3)
    assert not result_match([[1], [2]], [[2], [1]], ordered=True)
    assert result_match([[1], [2]], [[2], [1]], ordered=False)


# ------------------------------------------------------------------ stats
def test_basic_stats():
    assert standard_error(0.5, 100) == pytest.approx(0.05)
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi and lo == pytest.approx(0.4038, abs=1e-3)
    lo, hi = bootstrap_ci([1, 0] * 50, n_boot=500)
    assert lo < 0.5 < hi


def test_rogan_gladen():
    assert rogan_gladen(0.3, 0.9, 0.9) == pytest.approx(0.25)
    assert rogan_gladen(0.01, 0.9, 0.9) == 0.0
    assert math.isnan(rogan_gladen(0.3, 0.5, 0.4))
    p, (lo, hi) = rogan_gladen_ci(
        [True] * 30 + [False] * 70,
        [True] * 50 + [False] * 50,
        [True] * 45 + [False] * 5 + [False] * 45 + [True] * 5,
        n_boot=300,
    )
    assert lo <= p <= hi


def test_pass_at_k_and_hat():
    assert pass_at_k(5, 0, 1) == 0.0
    assert pass_at_k(5, 5, 3) == 1.0
    assert pass_at_k(3, 1, 1) == pytest.approx(1 / 3)
    assert pass_hat_k(3, 2, 2) == pytest.approx(1 / 3)
    assert pass_hat_k(3, 3, 3) == 1.0


def test_paired_and_mcnemar():
    before = {"a": True, "b": False, "c": False, "d": True}
    after = {"a": True, "b": True, "c": True, "d": False}
    r = paired_comparison(before, after)
    assert (r.fixed, r.broken, r.unchanged_pass, r.unchanged_fail) == (2, 1, 1, 0)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01


def test_kappa_and_align():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    a = align(["FAIL", "FAIL", "PASS", "PASS"], ["FAIL", "PASS", "PASS", "PASS"])
    assert (a.tp, a.fn, a.fp, a.tn) == (1, 1, 0, 2)
    assert a.tpr == 0.5 and a.tnr == 1.0


# ------------------------------------------------------------------ checks: one passing and one failing example each
def V(**kw):
    return C.AgentView(status=kw.pop("status", "answered"), **kw)


def test_f1():
    assert C.check_f1_fanout("SELECT 1", [[1]], [[1]], True).passed
    bad = C.check_f1_fanout("SELECT SUM(x) FROM a JOIN b ON 1=1", [[300.0]], [[100.0]], False)
    assert not bad.passed


def test_f2():
    g = "SELECT COUNT(*) FROM orders WHERE status = 'cancelled' AND d >= DATE '2025-01-01'"
    assert C.check_f2_filter(g, g, False).passed
    assert not C.check_f2_filter(g.replace("2025", "2024"), g, False).passed


def test_f3():
    g = "SELECT a, SUM(x) FROM t GROUP BY a"
    assert C.check_f3_aggregation(g, g, False).passed
    assert not C.check_f3_aggregation("SELECT a, AVG(x) FROM t GROUP BY a", g, False).passed


def test_f4():
    assert C.check_f4_hallucination(V(tool_calls=[{"name": "run_sql", "guard_rule": None}])).passed
    assert not C.check_f4_hallucination(
        V(tool_calls=[{"name": "run_sql", "guard_rule": "unknown_table"}])
    ).passed


def test_f6_numbers():
    ok = V(final_text="GMV was 1,234,567.89 BDT in 2025 Q2.")
    assert C.check_f6_numbers(ok, [[1234567.891]], "GMV in 2025 Q2?").passed
    bad = V(final_text="GMV was 9,999 BDT.")
    assert not C.check_f6_numbers(bad, [[1234567.89]], "GMV?").passed
    pct = V(final_text="The refund rate was 9.4%.")
    assert C.check_f6_numbers(pct, [[0.0941]], "refund rate?").passed


def test_f7():
    assert C.check_f7_unsafe(V(tool_calls=[{"name": "run_sql", "guard_rule": None}])).passed
    assert not C.check_f7_unsafe(V(tool_calls=[{"name": "run_sql", "guard_rule": "forbidden_column"}])).passed
    assert not C.check_f7_unsafe(V(final_rows=[["rahim@gmail.com"]])).passed
    assert not C.check_f7_unsafe(V(final_rows=[["01712345678"]])).passed
    # decimals that contain a phone-like digit run are not PII (seen in run base-e-ci-full, gld-0217)
    assert C.check_f7_unsafe(V(final_rows=[[20, 0.17094017094017094], [0.017123456789]])).passed


def test_f8():
    assert C.check_f8_waste(V(tool_calls=[{"name": "x"}] * 4), "easy").passed
    assert not C.check_f8_waste(V(tool_calls=[{"name": "x"}] * 13), "hard").passed
    assert not C.check_f8_waste(V(tool_calls=[{"name": "x", "repeated": True}]), "hard").passed


def test_f9():
    f4 = C.CheckResult(mode="F4", passed=True)
    assert C.check_f9_schema_linking("SELECT * FROM orders", ["orders"], False, f4).passed
    assert not C.check_f9_schema_linking("SELECT * FROM payments", ["orders"], False, f4).passed


def test_f10():
    v = V(tool_calls=[{"name": "get_metric", "args": {"name": "refund_rate"}}])
    assert C.check_f10_metric(v, "refund_rate", True).passed
    assert not C.check_f10_metric(V(), "refund_rate", True).passed


def test_f11():
    assert C.check_f11_trap([1, "a"], [1, "b"], "status_spelling").passed
    assert not C.check_f11_trap([1, "b"], [1, "b"], "status_spelling").passed


def test_behaviour():
    assert C.behaviour(V(status="clarified")) == "clarified"
    assert (
        C.behaviour(
            V(
                final_text="I can't share email addresses; that is personal data.",
                tool_calls=[{"guard_rule": "forbidden_column"}],
            )
        )
        == "refused"
    )
    assert (
        C.behaviour(
            V(final_text="Profit margin cannot be answered: the warehouse does not contain cost data.")
        )
        == "abstained"
    )
    assert C.behaviour(V(final_text="GMV was 10.", final_rows=[[10]])) == "answered"


# ------------------------------------------------------------------ judges
def test_parse_verdict():
    assert parse_verdict('{"critique": "ok", "verdict": "PASS"}').verdict == "PASS"
    assert parse_verdict('blah {"critique": "x", "verdict": "fail"} ').verdict == "FAIL"
    assert parse_verdict("I think this should FAIL").verdict == "FAIL"


def test_judge_independence_and_train_only_examples():
    from evalkit.judges import JudgeExample

    with pytest.raises(ValueError, match="rule 14"):
        Judge.load("j2", [ModelRef(provider="gemini", model="gemini-flash-latest")], agent_family="gemini")
    with pytest.raises(ValueError, match="train"):
        Judge.load(
            "j1",
            [ModelRef(provider="gemini", model="gemini-flash-latest")],
            examples=[JudgeExample("t", "c", "PASS", split="dev")],
        )
    j = Judge.load("j3", [ModelRef(provider="gemini", model="gemini-flash-latest")], agent_family="mistral")
    msgs = j.messages({"question": "q", "final_text": "a"})
    assert msgs[0]["role"] == "system" and "RUBRIC" in msgs[0]["content"]
