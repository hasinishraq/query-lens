"""Single-metric templates: 'What was <metric> in <period>?' with gold (metrics.yaml), alt and trap SQL.

Gold SQL is the official reference query from ``semantic/metrics.yaml``. The alt SQL is a separately written
formulation (different constructs) used for the double-written check. Trap SQL is the 'trap-ignored' variant.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

from querylens.semantic import load_metrics, render

from .common import ALL_PERIODS, DAYS, MONTHS, QUARTERS, RECENT, YEARS, Period

LABELS: dict[str, tuple[str, str, str]] = {
    "gmv": ("GMV (gross merchandise value)", "জিএমভি (মোট পণ্য বিক্রয় মূল্য)", "GMV"),
    "net_revenue": ("net revenue", "নিট রাজস্ব", "net revenue"),
    "order_count": ("the number of valid orders", "বৈধ অর্ডারের সংখ্যা", "valid order koto gula"),
    "aov": ("the average order value", "গড় অর্ডার মূল্য", "average order value"),
    "items_per_order": (
        "the average number of units per order",
        "প্রতি অর্ডারে গড় ইউনিট সংখ্যা",
        "order prati gore koto unit",
    ),
    "active_customers": ("the number of active customers", "সক্রিয় গ্রাহকের সংখ্যা", "active customer koto jon"),
    "new_customers": (
        "the number of new customer signups",
        "নতুন গ্রাহক নিবন্ধনের সংখ্যা",
        "notun customer signup koto",
    ),
    "repeat_rate": ("the repeat purchase rate", "পুনরায় ক্রয়ের হার", "repeat rate"),
    "cancellation_rate": ("the order cancellation rate", "অর্ডার বাতিলের হার", "order cancel er rate"),
    "refund_rate": ("the refund rate", "রিফান্ডের হার", "refund rate"),
    "return_rate": ("the item return rate", "পণ্য ফেরতের হার", "return rate"),
    "payment_success": ("the payment success rate", "পেমেন্ট সফলতার হার", "payment success rate"),
    "cod_share": ("the share of cash-on-delivery orders", "ক্যাশ অন ডেলিভারি অর্ডারের অংশ", "COD order er share"),
    "late_delivery_rate": ("the late delivery rate", "দেরিতে ডেলিভারির হার", "late delivery rate"),
    "avg_delivery_days": (
        "the average delivery time in days",
        "গড় ডেলিভারি সময় (দিন)",
        "gore delivery te koto din",
    ),
    "active_sellers": ("the number of active sellers", "সক্রিয় বিক্রেতার সংখ্যা", "active seller koto jon"),
    "seller_payout": ("the total net seller payout", "বিক্রেতাদের মোট নিট পেআউট", "seller payout total"),
    "promo_redemption_rate": ("the promo code redemption rate", "প্রোমো কোড ব্যবহারের হার", "promo use er rate"),
    "discount_cost": ("the total promotion discount given", "মোট প্রোমোশন ছাড়ের পরিমাণ", "total discount koto"),
    "session_conversion": (
        "the web session conversion rate",
        "ওয়েব সেশন কনভার্শন হার",
        "web session conversion rate",
    ),
    "avg_rating": ("the average product rating", "গড় পণ্য রেটিং", "average product rating"),
    "ticket_resolution_h": (
        "the average ticket resolution time in hours",
        "টিকিট সমাধানের গড় সময় (ঘণ্টা)",
        "ticket solve korte gore koto ghonta",
    ),
    "stockout_rate": ("the stockout rate", "স্টক শেষ হওয়ার হার", "stockout rate"),
    "points_redeemed": (
        "the number of loyalty points redeemed",
        "রিডিম করা লয়্যালটি পয়েন্টের সংখ্যা",
        "koto loyalty point redeem hoyeche",
    ),
}

DOMAINS: dict[str, list[str]] = {
    "gmv": ["orders", "reference"],
    "net_revenue": ["orders", "payments"],
    "order_count": ["orders"],
    "aov": ["orders", "reference"],
    "items_per_order": ["orders"],
    "active_customers": ["orders", "customers"],
    "new_customers": ["customers"],
    "repeat_rate": ["orders", "customers"],
    "cancellation_rate": ["orders"],
    "refund_rate": ["orders", "payments"],
    "return_rate": ["orders", "payments"],
    "payment_success": ["payments"],
    "cod_share": ["orders", "payments"],
    "late_delivery_rate": ["logistics", "orders"],
    "avg_delivery_days": ["logistics", "orders"],
    "active_sellers": ["sellers", "orders"],
    "seller_payout": ["sellers"],
    "promo_redemption_rate": ["marketing", "orders"],
    "discount_cost": ["marketing", "orders"],
    "session_conversion": ["web"],
    "avg_rating": ["support", "catalog"],
    "ticket_resolution_h": ["support"],
    "stockout_rate": ["inventory"],
    "points_redeemed": ["customers"],
}

RATE_METRICS = {
    "repeat_rate",
    "cancellation_rate",
    "refund_rate",
    "return_rate",
    "payment_success",
    "cod_share",
    "late_delivery_rate",
    "promo_redemption_rate",
    "session_conversion",
    "stockout_rate",
}
DIFFICULTY: dict[str, str] = {
    "gmv": "medium",
    "net_revenue": "medium",
    "order_count": "easy",
    "aov": "medium",
    "items_per_order": "medium",
    "active_customers": "easy",
    "new_customers": "easy",
    "repeat_rate": "medium",
    "cancellation_rate": "easy",
    "refund_rate": "medium",
    "return_rate": "medium",
    "payment_success": "medium",
    "cod_share": "easy",
    "late_delivery_rate": "medium",
    "avg_delivery_days": "medium",
    "active_sellers": "easy",
    "seller_payout": "easy",
    "promo_redemption_rate": "easy",
    "discount_cost": "easy",
    "session_conversion": "easy",
    "avg_rating": "easy",
    "ticket_resolution_h": "medium",
    "stockout_rate": "easy",
    "points_redeemed": "easy",
}

V_ALT = "o.is_deleted = FALSE AND o.status IN ('delivered', 'in_transit', 'returned')"


def _ts(p: Period) -> tuple[str, str]:
    nxt = p.end + timedelta(days=1)
    return f"TIMESTAMP '{p.s} 00:00:00'", f"TIMESTAMP '{nxt.isoformat()} 00:00:00'"


def _utc_ts(p: Period) -> tuple[str, str]:
    """Dhaka-day window expressed in UTC (Dhaka = UTC+6)."""
    nxt = p.end + timedelta(days=1)
    return (
        f"(TIMESTAMP '{p.s} 00:00:00' - INTERVAL 6 HOUR)",
        f"(TIMESTAMP '{nxt.isoformat()} 00:00:00' - INTERVAL 6 HOUR)",
    )


def _o(p: Period) -> str:
    a, b = _ts(p)
    return f"o.order_at >= {a} AND o.order_at < {b}"


def alt_sql(metric: str, p: Period) -> str:
    a, b = _ts(p)
    ua, ub = _utc_ts(p)
    ow = _o(p)
    valid_ids = f"(SELECT o.order_id FROM orders o WHERE {V_ALT} AND {ow})"
    m: dict[str, Callable[[], str]] = {
        "gmv": lambda: (
            f"""SELECT SUM(CASE WHEN o.currency = 'BDT' THEN o.total_bdt ELSE o.total_amount * (
            SELECT x.bdt_per_unit FROM currency_rates x WHERE x.rate_date = CAST(o.order_at AS DATE)
            AND x.currency = o.currency) END) AS v FROM orders o WHERE {V_ALT} AND {ow}"""
        ),
        "net_revenue": lambda: (
            f"""SELECT COALESCE((SELECT SUM(p.amount_bdt) FROM payments p WHERE p.status = 'captured'
            AND p.order_id IN {valid_ids}), 0) - COALESCE((SELECT SUM(r.amount_bdt) FROM refunds r
            WHERE r.order_id IN {valid_ids}), 0) AS v"""
        ),
        "order_count": lambda: f"SELECT COUNT(DISTINCT o.order_id) AS v FROM orders o WHERE {V_ALT} AND {ow}",
        "aov": lambda: (
            f"""SELECT SUM(t.v) / COUNT(*) AS v FROM (SELECT CASE WHEN o.currency = 'BDT' THEN o.total_bdt
            ELSE o.total_amount * x.bdt_per_unit END AS v FROM orders o LEFT JOIN currency_rates x
            ON x.rate_date = CAST(o.order_at AS DATE) AND x.currency = o.currency WHERE {V_ALT} AND {ow}) t"""
        ),
        "items_per_order": lambda: (
            f"""SELECT AVG(q) AS v FROM (SELECT oi.order_id, SUM(oi.quantity) AS q
            FROM order_items oi WHERE oi.order_id IN {valid_ids} GROUP BY oi.order_id) t"""
        ),
        "active_customers": lambda: (
            f"""SELECT COUNT(*) AS v FROM (SELECT DISTINCT o.customer_id FROM orders o
            WHERE {V_ALT} AND {ow}) t"""
        ),
        "new_customers": lambda: (
            f"""SELECT COUNT(c.customer_id) AS v FROM customers c WHERE c.is_deleted = FALSE
            AND c.signup_month >= DATE '{p.start.replace(day=1).isoformat()}' AND c.signup_month <= DATE '{p.e}'"""
        ),
        "repeat_rate": lambda: (
            f"""SELECT COUNT(*) FILTER (WHERE n >= 2) * 1.0 / COUNT(*) AS v FROM (
            SELECT o.customer_id, COUNT(*) AS n FROM orders o WHERE {V_ALT} AND {ow} GROUP BY 1) t"""
        ),
        "cancellation_rate": lambda: (
            f"""SELECT SUM(CASE WHEN o.status LIKE 'cancel%' THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
            AS v FROM orders o WHERE o.is_deleted = FALSE AND {ow}"""
        ),
        "refund_rate": lambda: (
            f"""WITH base AS (SELECT o.order_id FROM orders o WHERE o.is_deleted = FALSE AND {ow}
            AND EXISTS (SELECT 1 FROM payments p WHERE p.order_id = o.order_id AND p.status = 'captured'))
            SELECT AVG(CASE WHEN b.order_id IN (SELECT order_id FROM refunds) THEN 1.0 ELSE 0.0 END) AS v FROM base b"""
        ),
        "return_rate": lambda: (
            f"""SELECT (SELECT COUNT(DISTINCT r.order_item_id) FROM returns r JOIN order_items oi
            ON oi.order_item_id = r.order_item_id JOIN orders o ON o.order_id = oi.order_id WHERE o.is_deleted = FALSE
            AND o.status IN ('delivered', 'returned') AND {ow}) * 1.0 / (SELECT COUNT(*) FROM order_items oi
            JOIN orders o ON o.order_id = oi.order_id WHERE o.is_deleted = FALSE AND o.status IN ('delivered', 'returned')
            AND {ow}) AS v"""
        ),
        "payment_success": lambda: (
            f"""SELECT COUNT(*) FILTER (WHERE status = 'captured') * 1.0
            / COUNT(*) FILTER (WHERE status IN ('captured', 'failed')) AS v FROM payments
            WHERE created_at_utc >= {ua} AND created_at_utc < {ub}"""
        ),
        "cod_share": lambda: (
            f"""SELECT COUNT(*) FILTER (WHERE p.method = 'cod') * 1.0 / COUNT(*) AS v
            FROM payments p JOIN orders o ON o.order_id = p.order_id WHERE p.status = 'captured'
            AND o.is_deleted = FALSE AND {ow}"""
        ),
        "late_delivery_rate": lambda: (
            f"""SELECT SUM(CASE WHEN CAST(s.delivered_at_utc + INTERVAL 6 HOUR AS DATE)
            > s.promised_date THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS v FROM shipments s
            WHERE s.delivered_at_utc IS NOT NULL AND s.order_id IN (SELECT order_id FROM orders WHERE is_deleted = FALSE)
            AND s.delivered_at_utc >= {ua} AND s.delivered_at_utc < {ub}"""
        ),
        "avg_delivery_days": lambda: (
            f"""SELECT AVG(CAST(s.delivered_at_utc + INTERVAL 6 HOUR AS DATE)
            - CAST(s.shipped_at_utc + INTERVAL 6 HOUR AS DATE)) AS v FROM shipments s
            WHERE s.delivered_at_utc IS NOT NULL AND s.order_id IN (SELECT order_id FROM orders WHERE is_deleted = FALSE)
            AND s.delivered_at_utc >= {ua} AND s.delivered_at_utc < {ub}"""
        ),
        "active_sellers": lambda: (
            f"""SELECT COUNT(*) AS v FROM (SELECT DISTINCT oi.seller_id FROM order_items oi
            WHERE oi.order_id IN {valid_ids}) t"""
        ),
        "seller_payout": lambda: (
            f"""SELECT SUM(gross_bdt) - SUM(commission_bdt) AS v FROM seller_payouts
            WHERE period_start >= DATE '{p.start.replace(day=1).isoformat()}' AND period_start <= DATE '{p.e}'"""
        ),
        "promo_redemption_rate": lambda: (
            f"SELECT COUNT(o.promo_id) * 1.0 / COUNT(*) AS v FROM orders o WHERE {V_ALT} AND {ow}"
        ),
        "discount_cost": lambda: (
            f"""SELECT SUM(d) AS v FROM (SELECT oi.order_id, SUM(oi.discount_bdt) AS d
            FROM order_items oi WHERE oi.order_id IN {valid_ids} GROUP BY 1) t"""
        ),
        "session_conversion": lambda: (
            f"""SELECT COUNT(converted_order_id) * 1.0 / COUNT(*) AS v FROM web_sessions
            WHERE started_at_utc >= {ua} AND started_at_utc < {ub}"""
        ),
        "avg_rating": lambda: (
            f"""SELECT SUM(r.rating) * 1.0 / COUNT(*) AS v FROM product_reviews r
            WHERE r.product_id IN (SELECT product_id FROM products WHERE is_deleted = FALSE)
            AND r.created_at >= {a} AND r.created_at < {b}"""
        ),
        "ticket_resolution_h": lambda: (
            f"""SELECT AVG(epoch(resolved_at_utc) - epoch(created_at_utc)) / 3600.0 AS v
            FROM support_tickets WHERE resolved_at_utc IS NOT NULL AND created_at_utc >= {ua} AND created_at_utc < {ub}"""
        ),
        "stockout_rate": lambda: (
            f"""SELECT COUNT(*) FILTER (WHERE on_hand_units = 0) * 1.0 / COUNT(*) AS v
            FROM inventory_snapshots WHERE snapshot_date >= DATE '{p.s}' AND snapshot_date <= DATE '{p.e}'"""
        ),
        "points_redeemed": lambda: (
            f"""SELECT SUM(points) FILTER (WHERE txn_type = 'redeem') AS v FROM loyalty_transactions
            WHERE txn_at >= {a} AND txn_at < {b} AND txn_type = 'redeem'"""
        ),
    }
    return " ".join(m[metric]().split())


def gold_sql(metric: str, p: Period) -> str:
    return " ".join(render(load_metrics()[metric].sql, p.start, p.end).split())


# trap-ignored variants for the metrics that carry a data trap in single-metric questions
def trap_sql(metric: str, trap: str, p: Period) -> str | None:
    ow = p.between("CAST(o.order_at AS DATE)")
    if metric in ("gmv", "aov") and trap == "currency":
        agg = "SUM" if metric == "gmv" else "AVG"
        return (
            f"SELECT {agg}(o.total_amount) AS value FROM orders o WHERE NOT o.is_deleted AND o.status NOT IN "
            f"('cancelled', 'canceled', 'payment_failed') AND {ow}"
        )
    if metric == "cancellation_rate" and trap == "status_spelling":
        return (
            f"SELECT AVG(CASE WHEN o.status = 'cancelled' THEN 1.0 ELSE 0.0 END) AS value FROM orders o "
            f"WHERE NOT o.is_deleted AND {ow}"
        )
    if metric == "late_delivery_rate" and trap == "nulls":
        # NULL delivered_at counted as "not late" and kept in the denominator (shipments promised in the period)
        d = p.between("s.promised_date")
        return (
            f"SELECT AVG(CASE WHEN CAST(s.delivered_at_utc + INTERVAL 6 HOUR AS DATE) > s.promised_date THEN 1.0 "
            f"ELSE 0.0 END) AS value FROM shipments s WHERE {d}"
        )
    if metric in ("session_conversion", "payment_success") and trap == "timezone":
        col, tbl = (
            ("started_at_utc", "web_sessions")
            if metric == "session_conversion"
            else ("created_at_utc", "payments")
        )
        cond = p.between(f"CAST({col} AS DATE)")
        if metric == "session_conversion":
            return f"SELECT AVG(CASE WHEN converted_order_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS value FROM {tbl} WHERE {cond}"
        return (
            f"SELECT SUM(CASE WHEN status = 'captured' THEN 1 ELSE 0 END) * 1.0 / SUM(CASE WHEN status IN "
            f"('captured', 'failed') THEN 1 ELSE 0 END) AS value FROM {tbl} WHERE {cond}"
        )
    if metric == "net_revenue" and trap == "payment_retries":
        return (
            f"SELECT SUM(p.amount_bdt) AS value FROM payments p JOIN orders o ON o.order_id = p.order_id "
            f"WHERE NOT o.is_deleted AND o.status NOT IN ('cancelled', 'canceled', 'payment_failed') AND {ow}"
        )
    return None


PERIOD_POOL = {
    "default": MONTHS[6:] + QUARTERS + YEARS,
    "new_customers": MONTHS + QUARTERS,
    "stockout_rate": QUARTERS + YEARS + MONTHS[::3],
    "seller_payout": MONTHS + QUARTERS + YEARS,
    "all": ALL_PERIODS,
    "trap:nulls": RECENT,
    "trap:timezone": DAYS,
}
