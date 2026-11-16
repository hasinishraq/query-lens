import json

import duckdb
import pytest

from bazaar_world.check import check
from bazaar_world.generate import HASHES_PATH, build, table_hashes
from bazaar_world.schema import FOREIGN_KEYS, TABLES, target_rows, total_target


def test_schema_counts():
    assert len(TABLES) == 35
    assert sum(len(t.columns) for t in TABLES) == 266


def test_size_targets():
    assert total_target("full") == 995_786
    assert total_target("small") == 102_266


def test_built_db_counts(small_con):
    n_tables, n_cols = small_con.execute(
        "SELECT COUNT(DISTINCT table_name), COUNT(*) FROM information_schema.columns WHERE table_schema='main'"
    ).fetchone()
    assert (n_tables, n_cols) == (35, 266)
    for t in TABLES:
        n = small_con.execute(f'SELECT COUNT(*) FROM "{t.name}"').fetchone()[0]
        assert n == pytest.approx(target_rows(t.name, "small"), rel=0.02), t.name


def test_world_check_small(small_db):
    assert check(small_db, "small") == []


@pytest.mark.parametrize(("table", "col"), sorted(FOREIGN_KEYS))
def test_foreign_keys(small_con, table, col):
    ref_t, ref_c = FOREIGN_KEYS[(table, col)]
    orphans = small_con.execute(
        f'SELECT COUNT(*) FROM "{table}" x LEFT JOIN "{ref_t}" r ON r."{ref_c}" = x."{col}" '
        f'WHERE x."{col}" IS NOT NULL AND r."{ref_c}" IS NULL'
    ).fetchone()[0]
    assert orphans == 0


CONSISTENCY = {
    "bdt_total_matches_items": """
        SELECT COUNT(*) FROM orders o JOIN (SELECT order_id, ROUND(SUM(line_total_bdt), 2) s FROM order_items
        GROUP BY 1) i USING (order_id) WHERE o.currency = 'BDT' AND ABS(o.total_bdt - i.s) > 0.01""",
    "usd_total_bdt_null": "SELECT COUNT(*) FROM orders WHERE currency = 'USD' AND total_bdt IS NOT NULL",
    "captured_equals_total": """
        SELECT COUNT(*) FROM payments p JOIN orders o USING (order_id)
        WHERE p.status = 'captured' AND o.currency = 'BDT' AND ABS(p.amount_bdt - o.total_bdt) > 0.01""",
    "one_capture_per_order": """
        SELECT COUNT(*) FROM (SELECT order_id FROM payments WHERE status = 'captured'
        GROUP BY 1 HAVING COUNT(*) > 1)""",
    "refunds_le_captured": """
        SELECT COUNT(*) FROM (SELECT order_id, SUM(amount_bdt) r FROM refunds GROUP BY 1) r
        JOIN (SELECT order_id, SUM(amount_bdt) c FROM payments WHERE status = 'captured' GROUP BY 1) c
        USING (order_id) WHERE r > c + 0.01""",
    "no_shipments_for_cancelled": """
        SELECT COUNT(*) FROM shipments s JOIN orders o USING (order_id)
        WHERE o.status IN ('cancelled', 'canceled', 'payment_failed')""",
    "delivered_at_null_iff_undelivered": """
        SELECT COUNT(*) FROM shipments WHERE (status = 'delivered') <> (delivered_at_utc IS NOT NULL)""",
    "status_events_in_order": """
        SELECT COUNT(*) FROM (SELECT event_at_utc,
        LAG(event_at_utc) OVER (PARTITION BY order_id ORDER BY event_id) p FROM order_status_events)
        WHERE event_at_utc < p""",
    "shipment_events_in_order": """
        SELECT COUNT(*) FROM (SELECT event_at_utc, LAG(event_at_utc) OVER (PARTITION BY shipment_id
        ORDER BY shipment_event_id) p FROM shipment_events) WHERE event_at_utc < p""",
    "address_belongs_to_customer": """
        SELECT COUNT(*) FROM orders o JOIN customer_addresses a USING (address_id)
        WHERE a.customer_id <> o.customer_id""",
    "both_cancel_spellings": """
        SELECT 2 - COUNT(DISTINCT status) FROM orders WHERE status IN ('cancelled', 'canceled')""",
    "scd2_one_current_segment": """
        SELECT COUNT(*) FROM (SELECT customer_id FROM customer_segment_history WHERE is_current
        GROUP BY 1 HAVING COUNT(*) <> 1)""",
    "scd2_contiguous_prices": """
        SELECT COUNT(*) FROM (SELECT valid_from,
        LAG(valid_to) OVER (PARTITION BY variant_id ORDER BY valid_from) p FROM product_price_history)
        WHERE p IS NOT NULL AND p <> valid_from""",
    "redemption_iff_promo": """
        SELECT COUNT(*) FROM orders o FULL JOIN promotion_redemptions r USING (order_id)
        WHERE (o.promo_id IS NULL) <> (r.order_id IS NULL)""",
}


@pytest.mark.parametrize("name", sorted(CONSISTENCY))
def test_consistency(small_con, name):
    assert small_con.execute(CONSISTENCY[name]).fetchone()[0] == 0


def _q(con, sql):
    return con.execute(sql).fetchone()[0]


def test_distributions(small_con):
    app = _q(small_con, "SELECT AVG(CASE WHEN channel = 'app' THEN 1.0 ELSE 0 END) FROM orders")
    assert app == pytest.approx(0.62, abs=0.01)
    usd = _q(small_con, "SELECT AVG(CASE WHEN currency = 'USD' THEN 1.0 ELSE 0 END) FROM orders")
    assert usd == pytest.approx(0.012, abs=0.004)
    late = _q(
        small_con,
        """SELECT AVG(CASE WHEN CAST(delivered_at_utc + INTERVAL 6 HOUR AS DATE) > promised_date
              THEN 1.0 ELSE 0 END) FROM shipments WHERE status = 'delivered'""",
    )
    assert 0.09 < late < 0.16
    five = _q(small_con, "SELECT AVG(CASE WHEN rating = 5 THEN 1.0 ELSE 0 END) FROM product_reviews")
    assert five == pytest.approx(0.48, abs=0.01)
    once = _q(
        small_con,
        """WITH c AS (SELECT customer_id, COUNT(*) n FROM orders GROUP BY 1)
              SELECT AVG(CASE WHEN n = 1 THEN 1.0 ELSE 0 END) FROM c""",
    )
    assert once == pytest.approx(0.35, abs=0.02)


def test_determinism(tmp_path):
    """Two builds with seed 42 give identical table hashes (CLAUDE.md rule 7)."""
    hs = []
    for k in range(2):
        out = tmp_path / f"b{k}"
        build("small", 42, out, write_metadata=False)
        con = duckdb.connect(str(out / "bazaar_small.duckdb"), read_only=True)
        hs.append(table_hashes(con))
        con.close()
    assert hs[0] == hs[1]


def test_hashes_match_committed(small_con):
    ref = json.loads(HASHES_PATH.read_text(encoding="utf-8"))["small_seed42"]["tables"]
    assert table_hashes(small_con) == ref
