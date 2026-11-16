from datetime import date

import pytest
import sqlglot

from bazaar_world.mysqldb import fetchall, role_connect
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
    con = role_connect("small", "analyst")
    try:
        rows = fetchall(con, render(load_metrics()[name].sql, date(2025, 1, 1), date(2025, 12, 31)))
    finally:
        con.close()
    assert len(rows) == 1
    assert rows[0][0] is not None


@pytest.mark.full
@pytest.mark.parametrize("name", NAMES)
def test_metric_runs_full(full_db, name):
    con = role_connect("full", "analyst")
    try:
        rows = fetchall(con, render(load_metrics()[name].sql))
    finally:
        con.close()
    assert len(rows) == 1
    assert rows[0][0] is not None


def test_metric_tables_match_sql():
    for name, m in load_metrics().items():
        tree = sqlglot.parse_one(render(m.sql), read="mysql")
        ctes = {c.alias for c in tree.find_all(sqlglot.exp.CTE)}
        used = {t.name for t in tree.find_all(sqlglot.exp.Table)} - ctes
        assert used == set(m.tables), (name, used, m.tables)


def test_gmv_equals_aov_times_count(analyst_db):
    m = load_metrics()
    con = role_connect("small", "analyst")
    try:
        g = float(fetchall(con, render(m["gmv"].sql))[0][0])
        a = float(fetchall(con, render(m["aov"].sql))[0][0])
        n = float(fetchall(con, render(m["order_count"].sql))[0][0])
    finally:
        con.close()
    assert g == pytest.approx(a * n, rel=1e-9)


def test_render_only_replaces_params():
    sql = render("SELECT ':start_date_x', x FROM t WHERE d BETWEEN :start_date AND :end_date")
    assert "DATE '2024-01-01'" in sql
    assert "DATE '2026-10-31'" in sql
