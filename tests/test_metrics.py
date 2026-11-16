from datetime import date

import duckdb
import pytest
import sqlglot

from querylens.semantic import load_metrics, render

NAMES = [
    "gmv",
    "net_revenue",
    "order_count",
    "aov",
    "items_per_order",
    "active_customers",
    "new_customers",
    "repeat_rate",
    "cancellation_rate",
    "refund_rate",
    "return_rate",
    "payment_success",
    "cod_share",
    "late_delivery_rate",
    "avg_delivery_days",
    "active_sellers",
    "seller_payout",
    "promo_redemption_rate",
    "discount_cost",
    "session_conversion",
    "avg_rating",
    "ticket_resolution_h",
    "stockout_rate",
    "points_redeemed",
]


def test_metric_names():
    assert list(load_metrics()) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_metric_runs_small(analyst_db, name):
    con = duckdb.connect(str(analyst_db), read_only=True)
    try:
        rows = con.execute(render(load_metrics()[name].sql, date(2025, 1, 1), date(2025, 12, 31))).fetchall()
    finally:
        con.close()
    assert len(rows) == 1
    assert rows[0][0] is not None


@pytest.mark.full
@pytest.mark.parametrize("name", NAMES)
def test_metric_runs_full(full_db, name):
    con = duckdb.connect(str(full_db.parent / "bazaar_full__analyst.duckdb"), read_only=True)
    try:
        rows = con.execute(render(load_metrics()[name].sql)).fetchall()
    finally:
        con.close()
    assert len(rows) == 1
    assert rows[0][0] is not None


def test_metric_tables_match_sql():
    for name, m in load_metrics().items():
        tree = sqlglot.parse_one(render(m.sql), read="duckdb")
        ctes = {c.alias for c in tree.find_all(sqlglot.exp.CTE)}
        used = {t.name for t in tree.find_all(sqlglot.exp.Table)} - ctes
        assert used == set(m.tables), (name, used, m.tables)


def test_gmv_equals_aov_times_count(analyst_db):
    m = load_metrics()
    con = duckdb.connect(str(analyst_db), read_only=True)
    try:
        g = con.execute(render(m["gmv"].sql)).fetchone()[0]
        a = con.execute(render(m["aov"].sql)).fetchone()[0]
        n = con.execute(render(m["order_count"].sql)).fetchone()[0]
    finally:
        con.close()
    assert g == pytest.approx(a * n, rel=1e-9)


def test_render_only_replaces_params():
    sql = render("SELECT ':start_date_x', x FROM t WHERE d BETWEEN :start_date AND :end_date")
    assert "DATE '2024-01-01'" in sql
    assert "DATE '2026-10-31'" in sql
