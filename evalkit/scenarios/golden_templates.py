"""Golden question templates. Each template yields up to ``max_n`` distinct instances with gold, alt and
(for trap templates) trap SQL, plus English, Bangla and Banglish phrasings.

All SQL here is a *draft by Claude* (``verified_by: null``); the human verifies items in the viewer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np

from .common import (
    FISCAL,
    FX_JOIN,
    MONTHS,
    ORDER_VALUE,
    QUARTERS,
    RECENT,
    REGION_JOIN,
    VALID,
    VALID_ALT,
    YEARS,
    Instance,
    Period,
    bn,
    month,
    odate,
    odate_alt,
    pick,
    quarter,
)
from .metric_templates import (
    DIFFICULTY,
    DOMAINS,
    LABELS,
    PERIOD_POOL,
    RATE_METRICS,
    alt_sql,
    gold_sql,
    trap_sql,
)

Gen = Callable[["Ctx", int], list[Instance]]


@dataclass
class Ctx:
    rng: np.random.Generator
    divisions: list[str]
    couriers: list[str]
    warehouses: list[str]
    l1_categories: list[str]
    shared_sellers: list[str]
    usd_orders: list[int]
    skus: list[str]
    campaigns_acq: list[str]
    traffic_sources: list[str]
    tier_changers: list[str]
    depts_with_deleted: list[str]


@dataclass
class Template:
    id: str
    intent: str
    difficulty: str
    trap: str
    metric: str | None
    domains: list[str]
    max_n: int
    gen: Gen
    min_n: int = 0
    planted_fact: str | None = None
    tables_hint: str = ""


REGISTRY: list[Template] = []


def template(
    id: str,
    intent: str,
    difficulty: str,
    trap: str,
    metric: str | None,
    domains: list[str],
    max_n: int,
    min_n: int = 0,
    planted_fact: str | None = None,
) -> Callable[[Gen], Gen]:
    def deco(fn: Gen) -> Gen:
        REGISTRY.append(
            Template(id, intent, difficulty, trap, metric, domains, max_n, fn, min_n, planted_fact)
        )
        return fn

    return deco


def _clean(sql: str) -> str:
    return " ".join(sql.split())


def I(key: str, en: str, bn_: str, bl: str, gold: str, alt: str, trap: str | None = None, **kw) -> Instance:  # noqa: E743
    return Instance(
        key=key,
        en=en,
        bn=bn_,
        bl=bl,
        gold=_clean(gold),
        alt=_clean(alt),
        trap=_clean(trap) if trap else None,
        **kw,
    )


# =============================================================================== single-metric templates
def _metric_gen(metric: str, trap: str) -> Gen:
    def gen(ctx: Ctx, n: int) -> list[Instance]:
        pool = PERIOD_POOL.get(f"trap:{trap}") or PERIOD_POOL.get(metric, PERIOD_POOL["default"])
        out = []
        en_l, bn_l, bl_l = LABELS[metric]
        for p in pick(ctx.rng, pool, n):
            t = trap_sql(metric, trap, p) if trap != "none" else None
            out.append(
                I(
                    f"{metric}:{p.key}",
                    f"What was {en_l} in {p.en}?",
                    f"{p.bn} {bn_l} কত ছিল?",
                    f"{p.bl} {bl_l} koto chilo?",
                    gold_sql(metric, p),
                    alt_sql(metric, p),
                    t,
                )
            )
        return out

    return gen


METRIC_TRAPS = {
    "gmv": "currency",
    "aov": "currency",
    "cancellation_rate": "status_spelling",
    "late_delivery_rate": "nulls",
    "session_conversion": "timezone",
    "payment_success": "timezone",
    "net_revenue": "payment_retries",
}
for _m in LABELS:
    _intent = "ratio" if _m in RATE_METRICS else "aggregation"
    REGISTRY.append(
        Template(f"M-{_m}", _intent, DIFFICULTY[_m], "none", _m, DOMAINS[_m], 8, _metric_gen(_m, "none"))
    )
    if _m in METRIC_TRAPS:
        _tr = METRIC_TRAPS[_m]
        REGISTRY.append(
            Template(
                f"MT-{_m}",
                _intent,
                "medium" if DIFFICULTY[_m] == "easy" else "hard",
                _tr,
                _m,
                DOMAINS[_m] + (["reference"] if _tr == "currency" else []),
                8,
                _metric_gen(_m, _tr),
            )
        )


# =============================================================================== lookup
@template("LK-cancelled-count", "lookup", "easy", "status_spelling", None, ["orders"], 8)
def lk_cancelled(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS, n):
        out.append(
            I(
                p.key,
                f"How many orders were cancelled in {p.en}?",
                f"{p.bn} কতগুলো অর্ডার বাতিল হয়েছে?",
                f"{p.bl} koto gula order cancel hoyeche?",
                f"SELECT COUNT(*) AS n FROM orders o WHERE NOT o.is_deleted AND o.status IN ('cancelled', 'canceled') AND {odate(p)}",
                f"SELECT COUNT(order_id) AS n FROM orders o WHERE o.is_deleted = FALSE AND o.status LIKE 'cancel%' AND {odate_alt(p)}",
                f"SELECT COUNT(*) AS n FROM orders o WHERE NOT o.is_deleted AND o.status = 'cancelled' AND {odate(p)}",
            )
        )
    return out


@template("LK-seller-tier", "lookup", "medium", "scd2", None, ["sellers"], 6, min_n=3, planted_fact="F04")
def lk_tier(ctx: Ctx, n: int) -> list[Instance]:
    cases = [
        ("Meghna Mart", date(2025, 6, 30)),
        ("Meghna Mart", date(2025, 7, 1)),
        ("Meghna Mart", date(2024, 12, 31)),
    ]
    others = [s for s in ctx.tier_changers if s not in ("Meghna Mart", "Jamuna Fashion")]
    for s in pick(ctx.rng, others, max(0, n - 3)):
        cases.append((s, date(2025, int(ctx.rng.integers(1, 13)), 15)))
    out = []
    for s, d in cases[:n]:
        ds = d.isoformat()
        out.append(
            I(
                f"{s}:{ds}",
                f"Which tier was seller '{s}' on {ds}?",
                f"{bn(ds)} তারিখে বিক্রেতা '{s}' কোন টিয়ারে ছিল?",
                f"{ds} e seller '{s}' kon tier e chilo?",
                f"SELECT th.tier FROM seller_tier_history th JOIN sellers s ON s.seller_id = th.seller_id WHERE s.name = '{s}' AND th.valid_from <= DATE '{ds}' AND DATE '{ds}' < th.valid_to",
                f"SELECT tier FROM seller_tier_history WHERE seller_id IN (SELECT seller_id FROM sellers WHERE name = '{s}') AND DATE '{ds}' >= valid_from AND DATE '{ds}' < valid_to",
                f"SELECT th.tier FROM seller_tier_history th JOIN sellers s ON s.seller_id = th.seller_id WHERE s.name = '{s}' AND th.valid_to = DATE '9999-12-31'",
                planted_fact="F04" if s == "Meghna Mart" else None,
            )
        )
    return out


@template("LK-fx-rate", "lookup", "easy", "none", None, ["reference"], 5)
def lk_fx(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS, n):
        d = date(p.start.year, p.start.month, 10).isoformat()
        out.append(
            I(
                d,
                f"What was the USD to BDT exchange rate on {d}?",
                f"{bn(d)} তারিখে ডলার থেকে টাকার বিনিময় হার কত ছিল?",
                f"{d} e USD to BDT rate koto chilo?",
                f"SELECT bdt_per_unit FROM currency_rates WHERE currency = 'USD' AND rate_date = DATE '{d}'",
                f"SELECT MAX(bdt_per_unit) FROM currency_rates WHERE rate_date = CAST('{d}' AS DATE) AND currency = 'USD'",
            )
        )
    return out


@template("LK-products-in-dept", "lookup", "medium", "soft_delete", None, ["catalog"], 8)
def lk_products(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for c in pick(ctx.rng, ctx.depts_with_deleted, n):
        out.append(
            I(
                c,
                f"How many products are listed in the '{c}' department?",
                f"'{c}' বিভাগে কতগুলো পণ্য তালিকাভুক্ত আছে?",
                f"'{c}' department e koto gula product ache?",
                f"""SELECT COUNT(*) AS n FROM products p JOIN categories c3 ON c3.category_id = p.category_id
                         JOIN categories c2 ON c2.category_id = c3.parent_category_id JOIN categories c1 ON c1.category_id = c2.parent_category_id
                         WHERE c1.name = '{c}' AND NOT p.is_deleted""",
                f"""SELECT COUNT(product_id) AS n FROM products WHERE is_deleted = FALSE AND category_id IN (
                         SELECT category_id FROM categories WHERE level = 3 AND path LIKE '{c} > %')""",
                f"""SELECT COUNT(*) AS n FROM products p JOIN categories c3 ON c3.category_id = p.category_id
                         JOIN categories c2 ON c2.category_id = c3.parent_category_id JOIN categories c1 ON c1.category_id = c2.parent_category_id
                         WHERE c1.name = '{c}'""",
            )
        )
    return out


@template("LK-warehouse", "lookup", "easy", "none", None, ["inventory"], 6)
def lk_wh(ctx: Ctx, n: int) -> list[Instance]:
    out = [
        I(
            "active",
            "How many warehouses are currently active?",
            "বর্তমানে কতটি ওয়্যারহাউস চালু আছে?",
            "ekhon koyta warehouse active ache?",
            "SELECT COUNT(*) FROM warehouses WHERE is_active",
            "SELECT SUM(CASE WHEN is_active THEN 1 ELSE 0 END) FROM warehouses",
        )
    ]
    for w in pick(ctx.rng, ctx.warehouses, n - 1):
        out.append(
            I(
                w,
                f"What is the storage capacity of warehouse {w}?",
                f"{w} ওয়্যারহাউসের ধারণক্ষমতা কত?",
                f"{w} warehouse er capacity koto?",
                f"SELECT capacity_units FROM warehouses WHERE name = '{w}'",
                f"SELECT MAX(capacity_units) FROM warehouses WHERE name LIKE '{w}'",
            )
        )
    return out[:n]


@template("LK-sku-price", "lookup", "hard", "scd2", None, ["catalog"], 8)
def lk_sku(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for i, sku in enumerate(pick(ctx.rng, ctx.skus, n)):
        d = MONTHS[6 + 3 * i % 24].start.replace(day=20).isoformat()
        out.append(
            I(
                f"{sku}:{d}",
                f"What was the price in BDT of variant {sku} on {d}?",
                f"{bn(d)} তারিখে {sku} ভ্যারিয়েন্টের দাম কত টাকা ছিল?",
                f"{d} e {sku} variant er dam koto taka chilo?",
                f"""SELECT ph.price_bdt FROM product_price_history ph JOIN product_variants v ON v.variant_id = ph.variant_id
                         WHERE v.sku = '{sku}' AND ph.valid_from <= DATE '{d}' AND DATE '{d}' < ph.valid_to""",
                f"""SELECT price_bdt FROM product_price_history WHERE variant_id = (SELECT variant_id FROM product_variants
                         WHERE sku = '{sku}') AND DATE '{d}' BETWEEN valid_from AND valid_to - INTERVAL 1 DAY""",
                f"""SELECT ph.price_bdt FROM product_price_history ph JOIN product_variants v ON v.variant_id = ph.variant_id
                         WHERE v.sku = '{sku}' AND ph.valid_to = DATE '9999-12-31'""",
            )
        )
    return out


@template("LK-usd-order-total", "lookup", "medium", "currency", None, ["orders", "reference"], 6)
def lk_usd(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for oid in pick(ctx.rng, ctx.usd_orders, n):
        out.append(
            I(
                str(oid),
                f"What was the total of order {oid} in BDT?",
                f"অর্ডার {bn(oid)}-এর মোট মূল্য কত টাকা?",
                f"order {oid} er total koto taka?",
                f"SELECT {ORDER_VALUE} AS total_bdt FROM orders o {FX_JOIN} WHERE o.order_id = {oid}",
                f"""SELECT CASE WHEN currency = 'BDT' THEN total_bdt ELSE total_amount * (SELECT bdt_per_unit FROM currency_rates
                         WHERE currency = 'USD' AND rate_date = CAST(order_at AS DATE)) END FROM orders WHERE order_id = {oid}""",
                f"SELECT o.total_bdt FROM orders o WHERE o.order_id = {oid}",
            )
        )
    return out


@template(
    "LK-classic-panjabi",
    "lookup",
    "easy",
    "duplicate_names",
    None,
    ["catalog", "sellers"],
    6,
    min_n=3,
    planted_fact="F03",
)
def lk_dup(ctx: Ctx, n: int) -> list[Instance]:
    qs = [
        (
            "count",
            "How many different products are named 'Classic Panjabi'?",
            "'Classic Panjabi' নামে কতগুলো আলাদা পণ্য আছে?",
            "'Classic Panjabi' name e koyta alada product ache?",
            "SELECT COUNT(DISTINCT product_id) FROM products WHERE name = 'Classic Panjabi'",
            "SELECT COUNT(*) FROM (SELECT product_id FROM products WHERE name = 'Classic Panjabi' GROUP BY product_id)",
            "SELECT COUNT(DISTINCT name) FROM products WHERE name = 'Classic Panjabi'",
        ),
        (
            "sellers",
            "Which sellers sell a product called 'Classic Panjabi'?",
            "'Classic Panjabi' নামের পণ্য কোন কোন বিক্রেতা বিক্রি করে?",
            "'Classic Panjabi' product ta kon kon seller bikri kore?",
            "SELECT DISTINCT s.name FROM products p JOIN sellers s ON s.seller_id = p.seller_id WHERE p.name = 'Classic Panjabi'",
            "SELECT name FROM sellers WHERE seller_id IN (SELECT seller_id FROM products WHERE name = 'Classic Panjabi')",
            "SELECT s.name FROM products p JOIN sellers s ON s.seller_id = p.seller_id WHERE p.name = 'Classic Panjabi' LIMIT 1",
        ),
        (
            "prices",
            "List the list price of each product named 'Classic Panjabi' with its seller.",
            "'Classic Panjabi' নামের প্রতিটি পণ্যের তালিকা মূল্য ও বিক্রেতা দেখাও।",
            "'Classic Panjabi' name er protita product er list price ar seller dekhao.",
            "SELECT p.product_id, s.name, p.list_price_bdt FROM products p JOIN sellers s ON s.seller_id = p.seller_id WHERE p.name = 'Classic Panjabi'",
            "SELECT p.product_id, (SELECT name FROM sellers x WHERE x.seller_id = p.seller_id), p.list_price_bdt FROM products p WHERE p.name = 'Classic Panjabi'",
            "SELECT MIN(p.product_id), MIN(s.name), AVG(p.list_price_bdt) FROM products p JOIN sellers s ON s.seller_id = p.seller_id WHERE p.name = 'Classic Panjabi'",
        ),
        (
            "jamuna",
            "How many sellers are named 'Jamuna Fashion', and in which cities are they based?",
            "'Jamuna Fashion' নামে কতজন বিক্রেতা আছে এবং তারা কোন শহরে?",
            "'Jamuna Fashion' name e koyjon seller ache ar tara kon city te?",
            "SELECT s.seller_id, c.name FROM sellers s JOIN cities c ON c.city_id = s.city_id WHERE s.name = 'Jamuna Fashion'",
            "SELECT seller_id, (SELECT name FROM cities c WHERE c.city_id = s.city_id) FROM sellers s WHERE s.name = 'Jamuna Fashion'",
            "SELECT MIN(s.seller_id), MIN(c.name) FROM sellers s JOIN cities c ON c.city_id = s.city_id WHERE s.name = 'Jamuna Fashion'",
        ),
        (
            "units",
            "How many units of each product named 'Classic Panjabi' were sold in valid orders in 2025?",
            "২০২৫ সালে বৈধ অর্ডারে 'Classic Panjabi' নামের প্রতিটি পণ্যের কত ইউনিট বিক্রি হয়েছে?",
            "2025 e valid order e 'Classic Panjabi' name er protita product koto unit bikri hoyeche?",
            f"""SELECT p.product_id, SUM(oi.quantity) AS units FROM order_items oi JOIN product_variants v ON v.variant_id = oi.variant_id
             JOIN products p ON p.product_id = v.product_id JOIN orders o ON o.order_id = oi.order_id
             WHERE p.name = 'Classic Panjabi' AND {VALID} AND {YEARS[1].between("CAST(o.order_at AS DATE)")} GROUP BY p.product_id""",
            f"""SELECT v.product_id, SUM(oi.quantity) FROM order_items oi JOIN product_variants v USING (variant_id)
             WHERE v.product_id IN (SELECT product_id FROM products WHERE name = 'Classic Panjabi')
             AND oi.order_id IN (SELECT order_id FROM orders o WHERE {VALID_ALT} AND {odate_alt(YEARS[1])}) GROUP BY v.product_id""",
            f"""SELECT p.name, SUM(oi.quantity) FROM order_items oi JOIN product_variants v ON v.variant_id = oi.variant_id
             JOIN products p ON p.product_id = v.product_id JOIN orders o ON o.order_id = oi.order_id
             WHERE p.name = 'Classic Panjabi' AND {VALID} AND {YEARS[1].between("CAST(o.order_at AS DATE)")} GROUP BY p.name""",
        ),
        (
            "units24",
            "How many units of each 'Classic Panjabi' product (by product id) were sold in valid orders in 2024?",
            "২০২৪ সালে প্রতিটি 'Classic Panjabi' পণ্যের (product id অনুযায়ী) কত ইউনিট বিক্রি হয়েছে?",
            "2024 e protita 'Classic Panjabi' product (product id onujayi) koto unit bikri hoyeche?",
            f"""SELECT p.product_id, SUM(oi.quantity) AS units FROM order_items oi JOIN product_variants v ON v.variant_id = oi.variant_id
             JOIN products p ON p.product_id = v.product_id JOIN orders o ON o.order_id = oi.order_id
             WHERE p.name = 'Classic Panjabi' AND {VALID} AND {YEARS[0].between("CAST(o.order_at AS DATE)")} GROUP BY p.product_id""",
            f"""SELECT v.product_id, SUM(oi.quantity) FROM order_items oi JOIN product_variants v USING (variant_id)
             WHERE v.product_id IN (SELECT product_id FROM products WHERE name = 'Classic Panjabi')
             AND oi.order_id IN (SELECT order_id FROM orders o WHERE {VALID_ALT} AND {odate_alt(YEARS[0])}) GROUP BY v.product_id""",
            f"""SELECT p.name, SUM(oi.quantity) FROM order_items oi JOIN product_variants v ON v.variant_id = oi.variant_id
             JOIN products p ON p.product_id = v.product_id JOIN orders o ON o.order_id = oi.order_id
             WHERE p.name = 'Classic Panjabi' AND {VALID} AND {YEARS[0].between("CAST(o.order_at AS DATE)")} GROUP BY p.name""",
        ),
    ]
    out = []
    for k, en, b, bl, g, a, t in qs[:n]:
        out.append(I(k, en, b, bl, g, a, t, planted_fact="F03" if "Classic Panjabi" in en else None))
    return out


# =============================================================================== aggregation
@template(
    "AG-gmv-by-division",
    "aggregation",
    "medium",
    "soft_delete",
    "gmv",
    ["orders", "customers", "reference"],
    8,
)
def ag_gmv_div(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        out.append(
            I(
                p.key,
                f"What was GMV by division in {p.en}?",
                f"{p.bn} বিভাগভিত্তিক জিএমভি কত ছিল?",
                f"{p.bl} division onujayi GMV koto chilo?",
                f"SELECT r.name AS division, SUM({ORDER_VALUE}) AS gmv FROM orders o {FX_JOIN} {REGION_JOIN} WHERE {VALID} AND {odate(p)} GROUP BY r.name",
                f"""SELECT dv, SUM(v) FROM (SELECT (SELECT r.name FROM customer_addresses a JOIN cities c ON c.city_id = a.city_id JOIN regions r ON r.region_id = c.region_id WHERE a.address_id = o.address_id) AS dv,
                         CASE WHEN o.currency = 'BDT' THEN o.total_bdt ELSE o.total_amount * x.bdt_per_unit END AS v FROM orders o
                         LEFT JOIN currency_rates x ON x.rate_date = CAST(o.order_at AS DATE) AND x.currency = o.currency WHERE {VALID_ALT} AND {odate_alt(p)}) t GROUP BY dv""",
                f"SELECT r.name AS division, SUM({ORDER_VALUE}) AS gmv FROM orders o {FX_JOIN} {REGION_JOIN} WHERE o.status NOT IN ('cancelled', 'canceled', 'payment_failed') AND {odate(p)} GROUP BY r.name",
            )
        )
    return out


@template("AG-refunded-amount", "aggregation", "easy", "none", None, ["payments"], 6)
def ag_refunds(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[3:] + QUARTERS, n):
        out.append(
            I(
                p.key,
                f"What was the total amount refunded (by refund request date) in {p.en}?",
                f"{p.bn} (রিফান্ড অনুরোধের তারিখ অনুযায়ী) মোট কত টাকা রিফান্ড হয়েছে?",
                f"{p.bl} (request date onujayi) mot koto taka refund hoyeche?",
                f"SELECT SUM(amount_bdt) FROM refunds WHERE {p.between('CAST(requested_at AS DATE)')}",
                f"SELECT SUM(r.amount_bdt) FROM refunds r WHERE r.requested_at >= TIMESTAMP '{p.s}' AND r.requested_at < TIMESTAMP '{p.e}' + INTERVAL 1 DAY",
            )
        )
    return out


@template("AG-bkash-orders", "aggregation", "medium", "payment_retries", None, ["payments", "orders"], 9)
def ag_bkash(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    methods = ["bkash", "nagad", "card", "bank_transfer"]
    for i, p in enumerate(pick(ctx.rng, MONTHS[3:] + QUARTERS, n)):
        m = methods[i % len(methods)]
        out.append(
            I(
                f"{m}:{p.key}",
                f"How many orders were successfully paid with {m} in {p.en} (by order date)?",
                f"{p.bn} (অর্ডারের তারিখ অনুযায়ী) কতগুলো অর্ডার {m} দিয়ে সফলভাবে পরিশোধ হয়েছে?",
                f"{p.bl} (order date onujayi) koto gula order {m} diye successfully pay hoyeche?",
                f"SELECT COUNT(DISTINCT p.order_id) FROM payments p JOIN orders o ON o.order_id = p.order_id WHERE p.method = '{m}' AND p.status = 'captured' AND NOT o.is_deleted AND {odate(p)}",
                f"SELECT COUNT(*) FROM orders o WHERE o.is_deleted = FALSE AND {odate_alt(p)} AND EXISTS (SELECT 1 FROM payments x WHERE x.order_id = o.order_id AND x.method = '{m}' AND x.status = 'captured')",
                f"SELECT COUNT(*) FROM payments p JOIN orders o ON o.order_id = p.order_id WHERE p.method = '{m}' AND NOT o.is_deleted AND {odate(p)}",
            )
        )
    return out


@template("AG-usd-sales", "aggregation", "medium", "currency", None, ["orders", "reference"], 6)
def ag_usd(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        out.append(
            I(
                p.key,
                f"What was the BDT value of valid USD orders in {p.en}?",
                f"{p.bn} বৈধ ডলার অর্ডারগুলোর মূল্য কত টাকা?",
                f"{p.bl} valid USD order gulor value koto taka?",
                f"SELECT SUM(o.total_amount * cr.bdt_per_unit) FROM orders o {FX_JOIN} WHERE o.currency = 'USD' AND {VALID} AND {odate(p)}",
                f"SELECT SUM(o.total_amount * (SELECT bdt_per_unit FROM currency_rates x WHERE x.currency = 'USD' AND x.rate_date = CAST(o.order_at AS DATE))) FROM orders o WHERE o.currency = 'USD' AND {VALID_ALT} AND {odate_alt(p)}",
                f"SELECT SUM(o.total_amount) FROM orders o WHERE o.currency = 'USD' AND {VALID} AND {odate(p)}",
            )
        )
    return out


@template("AG-deliveries-per-courier", "aggregation", "easy", "none", None, ["logistics"], 6)
def ag_courier(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[6:], n):
        d = p.between("CAST(s.delivered_at_utc + INTERVAL 6 HOUR AS DATE)")
        out.append(
            I(
                p.key,
                f"How many shipments did each courier deliver in {p.en} (Dhaka dates)?",
                f"{p.bn} (ঢাকার তারিখ অনুযায়ী) প্রতিটি কুরিয়ার কতগুলো শিপমেন্ট ডেলিভারি করেছে?",
                f"{p.bl} (Dhaka date) protita courier koyta shipment deliver koreche?",
                f"SELECT c.name, COUNT(*) FROM shipments s JOIN couriers c ON c.courier_id = s.courier_id WHERE s.status = 'delivered' AND {d} GROUP BY c.name",
                f"SELECT (SELECT name FROM couriers c WHERE c.courier_id = s.courier_id), COUNT(s.shipment_id) FROM shipments s WHERE s.delivered_at_utc IS NOT NULL AND s.delivered_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND s.delivered_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR GROUP BY s.courier_id",
            )
        )
    return out


@template("AG-onhand", "aggregation", "easy", "none", None, ["inventory"], 6)
def ag_onhand(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for i, w in enumerate(pick(ctx.rng, ctx.warehouses[:11], n)):
        d = ["2026-10-25", "2026-06-07", "2025-12-28", "2025-09-07", "2025-03-02", "2024-11-03"][i % 6]
        out.append(
            I(
                f"{w}:{d}",
                f"How many units were on hand in total at warehouse {w} in the snapshot of {d}?",
                f"{bn(d)} তারিখের স্ন্যাপশটে {w} ওয়্যারহাউসে মোট কত ইউনিট মজুত ছিল?",
                f"{d} er snapshot e {w} warehouse e mot koto unit stock chilo?",
                f"SELECT SUM(s.on_hand_units) FROM inventory_snapshots s JOIN warehouses w ON w.warehouse_id = s.warehouse_id WHERE w.name = '{w}' AND s.snapshot_date = DATE '{d}'",
                f"SELECT SUM(on_hand_units) FROM inventory_snapshots WHERE warehouse_id = (SELECT warehouse_id FROM warehouses WHERE name = '{w}') AND snapshot_date = CAST('{d}' AS DATE)",
            )
        )
    return out


# =============================================================================== ranking
def _top_sellers_gmv(p: Period, k: int) -> tuple[str, str]:
    g = f"""SELECT s.seller_id, s.name, SUM(oi.line_total_bdt) AS gmv FROM order_items oi JOIN orders o ON o.order_id = oi.order_id
            JOIN sellers s ON s.seller_id = oi.seller_id WHERE {VALID} AND {odate(p)} GROUP BY s.seller_id, s.name ORDER BY gmv DESC LIMIT {k}"""
    a = f"""WITH t AS (SELECT oi.seller_id, SUM(oi.line_total_bdt) AS v FROM order_items oi WHERE oi.order_id IN
            (SELECT order_id FROM orders o WHERE {VALID_ALT} AND {odate_alt(p)}) GROUP BY 1)
            SELECT t.seller_id, s.name, t.v FROM t JOIN sellers s USING (seller_id) QUALIFY ROW_NUMBER() OVER (ORDER BY t.v DESC) <= {k}"""
    return g, a


@template("RK-top-sellers-gmv", "ranking", "medium", "none", "gmv", ["sellers", "orders"], 8)
def rk_sellers(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for i, p in enumerate(pick(ctx.rng, QUARTERS + YEARS + FISCAL, n)):
        k = [5, 3, 10][i % 3]
        g, a = _top_sellers_gmv(p, k)
        out.append(
            I(
                p.key,
                f"Who were the top {k} sellers by GMV in {p.en}?",
                f"{p.bn} জিএমভি অনুযায়ী শীর্ষ {bn(k)} বিক্রেতা কারা?",
                f"{p.bl} GMV onujayi top {k} seller kara?",
                g,
                a,
                ordered=True,
            )
        )
    return out


@template("RK-courier-late", "ranking", "hard", "nulls", "late_delivery_rate", ["logistics", "orders"], 8)
def rk_courier_late(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, RECENT, n):  # the nulls trap only bites while undelivered shipments exist
        d = p.between("CAST(s.delivered_at_utc + INTERVAL 6 HOUR AS DATE)")
        out.append(
            I(
                p.key,
                f"Which courier had the highest late delivery rate in {p.en}?",
                f"{p.bn} কোন কুরিয়ারের দেরিতে ডেলিভারির হার সবচেয়ে বেশি ছিল?",
                f"{p.bl} kon courier er late delivery rate sobcheye beshi chilo?",
                f"""SELECT c.name, AVG(CASE WHEN CAST(s.delivered_at_utc + INTERVAL 6 HOUR AS DATE) > s.promised_date THEN 1.0 ELSE 0.0 END) AS late_rate
                         FROM shipments s JOIN couriers c ON c.courier_id = s.courier_id JOIN orders o ON o.order_id = s.order_id
                         WHERE s.status = 'delivered' AND s.delivered_at_utc IS NOT NULL AND NOT o.is_deleted AND {d}
                         GROUP BY c.name ORDER BY late_rate DESC LIMIT 1""",
                f"""SELECT name, r FROM (SELECT c.name, SUM(CASE WHEN CAST(s.delivered_at_utc + INTERVAL 6 HOUR AS DATE) > s.promised_date THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS r
                         FROM shipments s JOIN couriers c USING (courier_id) WHERE s.delivered_at_utc IS NOT NULL AND s.order_id IN (SELECT order_id FROM orders WHERE is_deleted = FALSE)
                         AND {d} GROUP BY c.name) t ORDER BY r DESC LIMIT 1""",
                f"""SELECT c.name, AVG(CASE WHEN CAST(s.delivered_at_utc + INTERVAL 6 HOUR AS DATE) > s.promised_date THEN 1.0 ELSE 0.0 END) AS late_rate
                         FROM shipments s JOIN couriers c ON c.courier_id = s.courier_id WHERE {p.between("s.promised_date")}
                         GROUP BY c.name ORDER BY late_rate DESC LIMIT 1""",
                ordered=True,
            )
        )
    return out


@template("RK-dept-units", "ranking", "hard", "none", None, ["catalog", "orders"], 10)
def rk_dept(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        out.append(
            I(
                p.key,
                f"Which 3 departments (top-level categories) sold the most units in {p.en}?",
                f"{p.bn} কোন ৩টি বিভাগ (শীর্ষ স্তরের ক্যাটাগরি) সবচেয়ে বেশি ইউনিট বিক্রি করেছে?",
                f"{p.bl} kon 3 ta department sobcheye beshi unit bikri koreche?",
                f"""SELECT c1.name, SUM(oi.quantity) AS units FROM order_items oi JOIN orders o ON o.order_id = oi.order_id
                         JOIN product_variants v ON v.variant_id = oi.variant_id JOIN products p ON p.product_id = v.product_id
                         JOIN categories c3 ON c3.category_id = p.category_id JOIN categories c2 ON c2.category_id = c3.parent_category_id
                         JOIN categories c1 ON c1.category_id = c2.parent_category_id WHERE {VALID} AND {odate(p)}
                         GROUP BY c1.name ORDER BY units DESC LIMIT 3""",
                f"""SELECT split_part(c.path, ' > ', 1) AS dept, SUM(oi.quantity) AS u FROM order_items oi JOIN product_variants v USING (variant_id)
                         JOIN products p USING (product_id) JOIN categories c ON c.category_id = p.category_id
                         WHERE oi.order_id IN (SELECT order_id FROM orders o WHERE {VALID_ALT} AND {odate_alt(p)}) GROUP BY 1 ORDER BY u DESC LIMIT 3""",
                ordered=True,
            )
        )
    return out


@template(
    "RK-division-refund",
    "ranking",
    "hard",
    "none",
    "refund_rate",
    ["payments", "orders", "reference"],
    6,
    min_n=3,
    planted_fact="F02",
)
def rk_div_refund(ctx: Ctx, n: int) -> list[Instance]:
    months = [month(2026, 3), month(2026, 3), month(2026, 3)] + pick(
        ctx.rng, [m for m in MONTHS[12:] if m.key != "2026-03"], 3
    )
    phr = [
        "Which division had the highest refund rate in {p}?",
        "Rank divisions by refund rate for {p}.",
        "Which division stood out with the highest refund rate in {p}, and what was the rate?",
    ]
    out = []
    for i, p in enumerate(months[:n]):
        g = f"""WITH cap AS (SELECT DISTINCT order_id FROM payments WHERE status = 'captured'), ref AS (SELECT DISTINCT order_id FROM refunds)
                SELECT r.name, AVG(CASE WHEN ref.order_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS refund_rate FROM orders o
                JOIN cap ON cap.order_id = o.order_id {REGION_JOIN} LEFT JOIN ref ON ref.order_id = o.order_id
                WHERE NOT o.is_deleted AND {odate(p)} GROUP BY r.name ORDER BY refund_rate DESC"""
        a = f"""SELECT dv, COUNT(*) FILTER (WHERE has_ref) * 1.0 / COUNT(*) AS rr FROM (SELECT r.name AS dv,
                EXISTS (SELECT 1 FROM refunds x WHERE x.order_id = o.order_id) AS has_ref FROM orders o
                JOIN customer_addresses a USING (address_id) JOIN cities c ON c.city_id = a.city_id JOIN regions r ON r.region_id = c.region_id
                WHERE o.is_deleted = FALSE AND {odate_alt(p)} AND o.order_id IN (SELECT order_id FROM payments WHERE status = 'captured')) t
                GROUP BY dv ORDER BY rr DESC"""
        if i % 3 != 1:
            g += " LIMIT 1"
            a += " LIMIT 1"
        en = phr[i % 3].format(p=p.en)
        out.append(
            I(
                f"{p.key}:{i}",
                en,
                f"{p.bn} কোন বিভাগে রিফান্ডের হার সবচেয়ে বেশি ছিল?",
                f"{p.bl} kon division e refund rate sobcheye beshi chilo?",
                g,
                a,
                ordered=True,
                planted_fact="F02" if p.key == "2026-03" else None,
                must_mention=["Chattogram"] if p.key == "2026-03" else [],
            )
        )
    return out


@template(
    "RK-warehouse-stockout",
    "ranking",
    "medium",
    "none",
    "stockout_rate",
    ["inventory"],
    6,
    min_n=3,
    planted_fact="F08",
)
def rk_stockout(ctx: Ctx, n: int) -> list[Instance]:
    ps = [YEARS[2], YEARS[2], YEARS[2], YEARS[1], YEARS[0], quarter(2026, 2)]
    phr = [
        "Which warehouse had the highest stockout rate in {p}?",
        "Rank warehouses by stockout rate in {p}.",
        "Which warehouse ran out of stock most often in {p}?",
    ]
    out = []
    for i, p in enumerate(ps[:n]):
        g = f"""SELECT w.name, AVG(CASE WHEN s.on_hand_units = 0 THEN 1.0 ELSE 0.0 END) AS stockout_rate FROM inventory_snapshots s
                JOIN warehouses w ON w.warehouse_id = s.warehouse_id WHERE {p.between("s.snapshot_date")} GROUP BY w.name ORDER BY stockout_rate DESC"""
        a = f"""SELECT (SELECT name FROM warehouses w WHERE w.warehouse_id = s.warehouse_id) AS wn, COUNT(*) FILTER (WHERE on_hand_units = 0) * 1.0 / COUNT(*) AS r
                FROM inventory_snapshots s WHERE s.snapshot_date >= DATE '{p.s}' AND s.snapshot_date <= DATE '{p.e}' GROUP BY s.warehouse_id ORDER BY r DESC"""
        if i % 3 != 1:
            g += " LIMIT 1"
            a += " LIMIT 1"
        out.append(
            I(
                f"{p.key}:{i}",
                phr[i % 3].format(p=p.en),
                f"{p.bn} কোন ওয়্যারহাউসে স্টক শেষ হওয়ার হার সবচেয়ে বেশি ছিল?",
                f"{p.bl} kon warehouse e stockout rate sobcheye beshi chilo?",
                g,
                a,
                ordered=True,
                planted_fact="F08" if p.key == "2026" else None,
                must_mention=["Gazipur-2"] if p.key == "2026" else [],
            )
        )
    return out


@template(
    "RK-top-districts", "ranking", "medium", "soft_delete", None, ["orders", "customers", "reference"], 8
)
def rk_districts(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        out.append(
            I(
                p.key,
                f"Which 5 districts had the most valid orders in {p.en}?",
                f"{p.bn} কোন ৫টি জেলায় সবচেয়ে বেশি বৈধ অর্ডার হয়েছে?",
                f"{p.bl} kon 5 ta district e sobcheye beshi valid order hoyeche?",
                f"""SELECT ci.name, COUNT(*) AS orders FROM orders o JOIN customer_addresses a ON a.address_id = o.address_id
                         JOIN cities ci ON ci.city_id = a.city_id WHERE {VALID} AND {odate(p)} GROUP BY ci.name ORDER BY orders DESC LIMIT 5""",
                f"""SELECT d, n FROM (SELECT ci.name AS d, COUNT(o.order_id) AS n, RANK() OVER (ORDER BY COUNT(o.order_id) DESC) AS rk FROM orders o
                         JOIN customer_addresses a USING (address_id) JOIN cities ci ON ci.city_id = a.city_id WHERE {VALID_ALT} AND {odate_alt(p)} GROUP BY ci.name) t
                         WHERE rk <= 5 ORDER BY n DESC""",
                f"""SELECT ci.name, COUNT(*) AS orders FROM orders o JOIN customer_addresses a ON a.address_id = o.address_id
                         JOIN cities ci ON ci.city_id = a.city_id WHERE o.status NOT IN ('cancelled', 'canceled', 'payment_failed') AND {odate(p)}
                         GROUP BY ci.name ORDER BY orders DESC LIMIT 5""",
                ordered=True,
            )
        )
    return out


F01_SQL = """WITH ss AS (SELECT DISTINCT order_id, seller_id FROM order_items),
    s AS (SELECT sh.shipment_id, sh.order_id, sh.promised_date, CAST(sh.delivered_at_utc + INTERVAL 6 HOUR AS DATE) AS dd
          FROM shipments sh JOIN orders o ON o.order_id = sh.order_id WHERE sh.status = 'delivered' AND sh.delivered_at_utc IS NOT NULL AND NOT o.is_deleted)
    SELECT se.name, COUNT(DISTINCT s.shipment_id) AS n, AVG(CASE WHEN s.dd > s.promised_date THEN 1.0 ELSE 0.0 END) AS late_rate
    FROM s JOIN ss ON ss.order_id = s.order_id JOIN sellers se ON se.seller_id = ss.seller_id
    WHERE s.dd BETWEEN DATE '2026-07-01' AND DATE '2026-09-30' GROUP BY se.seller_id, se.name HAVING COUNT(DISTINCT s.shipment_id) >= {k}
    ORDER BY late_rate DESC LIMIT 1"""
F01_ALT = """SELECT se.name, t.n, t.r FROM (SELECT x.seller_id, COUNT(*) AS n, SUM(x.late) * 1.0 / COUNT(*) AS r FROM (
    SELECT DISTINCT oi.seller_id, sh.shipment_id, CASE WHEN CAST(sh.delivered_at_utc + INTERVAL 6 HOUR AS DATE) > sh.promised_date THEN 1 ELSE 0 END AS late
    FROM shipments sh JOIN order_items oi ON oi.order_id = sh.order_id WHERE sh.delivered_at_utc IS NOT NULL
    AND sh.order_id IN (SELECT order_id FROM orders WHERE is_deleted = FALSE)
    AND sh.delivered_at_utc >= TIMESTAMP '2026-07-01' - INTERVAL 6 HOUR AND sh.delivered_at_utc < TIMESTAMP '2026-10-01' - INTERVAL 6 HOUR) x
    GROUP BY x.seller_id HAVING COUNT(*) >= {k}) t JOIN sellers se ON se.seller_id = t.seller_id ORDER BY t.r DESC LIMIT 1"""


@template(
    "RK-F01-seller-late",
    "ranking",
    "extra_hard",
    "none",
    "late_delivery_rate",
    ["sellers", "logistics", "orders"],
    3,
    min_n=3,
    planted_fact="F01",
)
def rk_f01(ctx: Ctx, n: int) -> list[Instance]:
    qs = [
        (
            "Which seller had the highest late delivery rate in 2026 Q3 among sellers with at least 300 delivered shipments?",
            300,
        ),
        (
            "Among sellers with 300 or more delivered shipments in July-September 2026, who was late most often?",
            300,
        ),
        (
            "Which seller with at least 200 delivered shipments had the worst late-delivery rate in Q3 2026, and what was it?",
            200,
        ),
    ]
    out = []
    for en, k in qs[:n]:
        out.append(
            I(
                f"{k}:{en[:20]}",
                en,
                f"২০২৬ সালের তৃতীয় প্রান্তিকে কমপক্ষে {bn(k)}টি ডেলিভারি হওয়া বিক্রেতাদের মধ্যে কার দেরিতে ডেলিভারির হার সবচেয়ে বেশি?",
                f"2026 Q3 te kompokkhe {k} ta delivered shipment ache emon seller der moddhe kar late delivery rate sobcheye beshi?",
                F01_SQL.format(k=k),
                F01_ALT.format(k=k),
                ordered=True,
                planted_fact="F01",
                must_mention=["Padma Gadgets"],
            )
        )
    return out


@template("RK-brand-rating", "ranking", "medium", "none", "avg_rating", ["support", "catalog"], 6)
def rk_brand(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, YEARS + FISCAL, n):
        out.append(
            I(
                p.key,
                f"Which 5 brands had the highest average rating in {p.en} (brands with at least 20 reviews)?",
                f"{p.bn} (কমপক্ষে ২০টি রিভিউ আছে এমন) কোন ৫টি ব্র্যান্ডের গড় রেটিং সবচেয়ে বেশি?",
                f"{p.bl} (kompokkhe 20 ta review ache emon) kon 5 ta brand er average rating sobcheye beshi?",
                f"""SELECT b.name, AVG(r.rating) AS avg_rating FROM product_reviews r JOIN products p ON p.product_id = r.product_id
                         JOIN brands b ON b.brand_id = p.brand_id WHERE NOT p.is_deleted AND {p.between("CAST(r.created_at AS DATE)")}
                         GROUP BY b.brand_id, b.name HAVING COUNT(*) >= 20 ORDER BY avg_rating DESC, b.name LIMIT 5""",
                f"""WITH t AS (SELECT p.brand_id, SUM(r.rating) * 1.0 / COUNT(*) AS a, COUNT(*) AS n FROM product_reviews r JOIN products p USING (product_id)
                         WHERE p.is_deleted = FALSE AND r.created_at >= TIMESTAMP '{p.s}' AND r.created_at < TIMESTAMP '{p.e}' + INTERVAL 1 DAY GROUP BY 1)
                         SELECT b.name, t.a FROM t JOIN brands b USING (brand_id) WHERE t.n >= 20 ORDER BY t.a DESC, b.name LIMIT 5""",
                ordered=True,
            )
        )
    return out


@template("RK-method-success", "ranking", "medium", "none", "payment_success", ["payments"], 6)
def rk_method(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        d = p.between("CAST(created_at_utc + INTERVAL 6 HOUR AS DATE)")
        out.append(
            I(
                p.key,
                f"Which payment method had the lowest payment success rate in {p.en}?",
                f"{p.bn} কোন পেমেন্ট পদ্ধতির সফলতার হার সবচেয়ে কম ছিল?",
                f"{p.bl} kon payment method er success rate sobcheye kom chilo?",
                f"""SELECT method, SUM(CASE WHEN status = 'captured' THEN 1 ELSE 0 END) * 1.0 / SUM(CASE WHEN status IN ('captured', 'failed') THEN 1 ELSE 0 END) AS success
                         FROM payments WHERE {d} AND method <> 'cod' GROUP BY method ORDER BY success ASC LIMIT 1""",
                f"""SELECT method, COUNT(*) FILTER (WHERE status = 'captured') * 1.0 / COUNT(*) FILTER (WHERE status IN ('captured', 'failed')) AS s
                         FROM payments WHERE created_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND created_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR
                         AND method IN ('bkash', 'nagad', 'card', 'bank_transfer') GROUP BY method ORDER BY s LIMIT 1""",
                ordered=True,
            )
        )
    return out


@template("RK-top-products", "ranking", "hard", "duplicate_names", "gmv", ["catalog", "orders"], 6)
def rk_products(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    dup = "p.name IN (SELECT name FROM products GROUP BY name HAVING COUNT(*) > 1)"
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        base = f"""FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN product_variants v ON v.variant_id = oi.variant_id
                   JOIN products p ON p.product_id = v.product_id WHERE {VALID} AND {odate(p)} AND {dup}"""
        out.append(
            I(
                p.key,
                f"Some product names are shared by more than one product. Among those products (by product id), which 5 had the highest GMV in {p.en}?",
                f"কিছু পণ্যের নাম একাধিক পণ্যে ব্যবহৃত। {p.bn} সেই পণ্যগুলোর (product id অনুযায়ী) মধ্যে জিএমভিতে শীর্ষ ৫টি কোনগুলো?",
                f"kichu product er name ekadhik product e use hoy. {p.bl} oi product gulor (product id onujayi) moddhe GMV te top 5 konggula?",
                f"SELECT p.product_id, p.name, SUM(oi.line_total_bdt) AS gmv {base} GROUP BY p.product_id, p.name ORDER BY gmv DESC, p.product_id LIMIT 5",
                f"""SELECT v.product_id, (SELECT name FROM products x WHERE x.product_id = v.product_id), SUM(oi.line_total_bdt) AS g FROM order_items oi
                         JOIN product_variants v USING (variant_id) WHERE oi.order_id IN (SELECT order_id FROM orders o WHERE {VALID_ALT} AND {odate_alt(p)})
                         AND v.product_id IN (SELECT product_id FROM products WHERE name IN (SELECT name FROM products GROUP BY 1 HAVING COUNT(*) >= 2))
                         GROUP BY v.product_id ORDER BY g DESC, v.product_id LIMIT 5""",
                f"SELECT MIN(p.product_id), p.name, SUM(oi.line_total_bdt) AS gmv {base} GROUP BY p.name ORDER BY gmv DESC LIMIT 5",
                ordered=True,
            )
        )
    return out


@template("RK-top-sellers-orders", "ranking", "medium", "duplicate_names", None, ["sellers", "orders"], 6)
def rk_sellers_orders(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, YEARS + FISCAL + QUARTERS[-4:], n):
        base = (
            f"FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN sellers s ON s.seller_id = oi.seller_id "
            f"WHERE {VALID} AND {odate(p)} AND s.name = 'Jamuna Fashion'"
        )
        out.append(
            I(
                p.key,
                f"Two different sellers are called 'Jamuna Fashion'. Rank them (by seller id) by the number of valid orders they received in {p.en}.",
                f"'Jamuna Fashion' নামে দুজন আলাদা বিক্রেতা আছে। {p.bn} তারা কতগুলো বৈধ অর্ডার পেয়েছে সেই অনুযায়ী (seller id সহ) সাজাও।",
                f"'Jamuna Fashion' name e duijon alada seller ache. {p.bl} tara koyta valid order peyeche sei onujayi (seller id soho) rank koro.",
                f"SELECT s.seller_id, s.name, COUNT(DISTINCT o.order_id) AS orders {base} GROUP BY s.seller_id, s.name ORDER BY orders DESC, s.seller_id",
                f"""SELECT x.seller_id, 'Jamuna Fashion', COUNT(*) AS c FROM (SELECT DISTINCT oi.seller_id, oi.order_id FROM order_items oi
                         WHERE oi.seller_id IN (SELECT seller_id FROM sellers WHERE name = 'Jamuna Fashion')
                         AND oi.order_id IN (SELECT order_id FROM orders o WHERE {VALID_ALT} AND {odate_alt(p)})) x GROUP BY x.seller_id ORDER BY c DESC, x.seller_id""",
                f"SELECT MIN(s.seller_id), s.name, COUNT(DISTINCT o.order_id) AS orders {base} GROUP BY s.name",
                ordered=True,
            )
        )
    return out


@template("RK-traffic-conversion", "ranking", "medium", "none", "session_conversion", ["web"], 6)
def rk_traffic(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        d = p.between("CAST(started_at_utc + INTERVAL 6 HOUR AS DATE)")
        out.append(
            I(
                p.key,
                f"Rank web traffic sources by session conversion rate in {p.en}.",
                f"{p.bn} সেশন কনভার্শন হার অনুযায়ী ওয়েব ট্রাফিক সোর্সগুলো সাজাও।",
                f"{p.bl} session conversion rate onujayi traffic source gula rank koro.",
                f"SELECT traffic_source, AVG(CASE WHEN converted_order_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS conv FROM web_sessions WHERE {d} GROUP BY 1 ORDER BY conv DESC",
                f"SELECT traffic_source, COUNT(converted_order_id) * 1.0 / COUNT(*) AS c FROM web_sessions WHERE started_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND started_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR GROUP BY traffic_source ORDER BY c DESC",
                ordered=True,
            )
        )
    return out


# =============================================================================== trend
@template("TR-monthly-gmv", "trend", "medium", "none", "gmv", ["orders", "reference"], 8)
def tr_gmv(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, YEARS + FISCAL + QUARTERS[-3:], n):
        out.append(
            I(
                p.key,
                f"Show monthly GMV for {p.en}.",
                f"{p.bn} মাসভিত্তিক জিএমভি দেখাও।",
                f"{p.bl} month onujayi GMV dekhao.",
                f"SELECT date_trunc('month', o.order_at) AS month, SUM({ORDER_VALUE}) AS gmv FROM orders o {FX_JOIN} WHERE {VALID} AND {odate(p)} GROUP BY 1 ORDER BY 1",
                f"""SELECT CAST(strftime(o.order_at, '%Y-%m-01') AS DATE) AS m, SUM(CASE WHEN o.currency = 'BDT' THEN o.total_bdt ELSE o.total_amount * x.bdt_per_unit END)
                         FROM orders o LEFT JOIN currency_rates x ON x.rate_date = CAST(o.order_at AS DATE) AND x.currency = o.currency WHERE {VALID_ALT} AND {odate_alt(p)} GROUP BY 1 ORDER BY 1""",
                ordered=True,
            )
        )
    return out


@template("TR-daily-captured", "trend", "medium", "timezone", None, ["payments"], 10)
def tr_captured(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[3:], n):
        out.append(
            I(
                p.key,
                f"How many payments were captured on each day of {p.en} (Dhaka dates)?",
                f"{p.bn} প্রতিদিন (ঢাকার তারিখ অনুযায়ী) কতগুলো পেমেন্ট সফল হয়েছে?",
                f"{p.bl} protidin (Dhaka date) koyta payment captured hoyeche?",
                f"SELECT CAST(captured_at_utc + INTERVAL 6 HOUR AS DATE) AS day, COUNT(*) FROM payments WHERE status = 'captured' AND {p.between('CAST(captured_at_utc + INTERVAL 6 HOUR AS DATE)')} GROUP BY 1 ORDER BY 1",
                f"SELECT CAST(captured_at_utc + INTERVAL '6 hours' AS DATE) AS d, COUNT(payment_id) FROM payments WHERE status = 'captured' AND captured_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND captured_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR GROUP BY 1 ORDER BY 1",
                f"SELECT CAST(captured_at_utc AS DATE) AS day, COUNT(*) FROM payments WHERE status = 'captured' AND {p.between('CAST(captured_at_utc AS DATE)')} GROUP BY 1 ORDER BY 1",
                ordered=True,
            )
        )
    return out


@template("TR-monthly-cancellation", "trend", "medium", "status_spelling", "cancellation_rate", ["orders"], 8)
def tr_cancel(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, YEARS + FISCAL + QUARTERS[-4:], n):
        g = f"SELECT date_trunc('month', o.order_at) AS month, AVG(CASE WHEN o.status IN ('cancelled', 'canceled') THEN 1.0 ELSE 0.0 END) AS rate FROM orders o WHERE NOT o.is_deleted AND {odate(p)} GROUP BY 1 ORDER BY 1"
        out.append(
            I(
                p.key,
                f"What was the monthly order cancellation rate in {p.en}?",
                f"{p.bn} মাসভিত্তিক অর্ডার বাতিলের হার কত ছিল?",
                f"{p.bl} month onujayi order cancel rate koto chilo?",
                g,
                f"SELECT CAST(strftime(o.order_at, '%Y-%m-01') AS DATE), SUM(CASE WHEN o.status LIKE 'cancel%' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) FROM orders o WHERE o.is_deleted = FALSE AND {odate_alt(p)} GROUP BY 1 ORDER BY 1",
                g.replace("IN ('cancelled', 'canceled')", "= 'cancelled'"),
                ordered=True,
            )
        )
    return out


@template("TR-monthly-new-customers", "trend", "easy", "none", "new_customers", ["customers"], 6)
def tr_newcust(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, YEARS + FISCAL + QUARTERS[-3:], n):
        out.append(
            I(
                p.key,
                f"How many new customers signed up each month in {p.en}?",
                f"{p.bn} প্রতি মাসে কতজন নতুন গ্রাহক নিবন্ধন করেছে?",
                f"{p.bl} protimashe koto jon notun customer signup koreche?",
                f"SELECT signup_month, COUNT(*) FROM customers WHERE NOT is_deleted AND {p.between('signup_month')} GROUP BY 1 ORDER BY 1",
                f"SELECT c.signup_month, COUNT(c.customer_id) FROM customers c WHERE c.is_deleted = FALSE AND c.signup_month >= DATE '{p.s}' AND c.signup_month <= DATE '{p.e}' GROUP BY c.signup_month ORDER BY 1",
                ordered=True,
            )
        )
    return out


@template("TR-daily-conversion", "trend", "medium", "timezone", "session_conversion", ["web"], 8)
def tr_conv(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[6:], n):
        d = "CAST(started_at_utc + INTERVAL 6 HOUR AS DATE)"
        out.append(
            I(
                p.key,
                f"Show the daily web session conversion rate for {p.en} (Dhaka dates).",
                f"{p.bn} (ঢাকার তারিখ অনুযায়ী) দৈনিক ওয়েব সেশন কনভার্শন হার দেখাও।",
                f"{p.bl} (Dhaka date) protidin er web session conversion rate dekhao.",
                f"SELECT {d} AS day, AVG(CASE WHEN converted_order_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS conv FROM web_sessions WHERE {p.between(d)} GROUP BY 1 ORDER BY 1",
                f"SELECT CAST(started_at_utc + INTERVAL '6 hours' AS DATE) AS dd, COUNT(converted_order_id) * 1.0 / COUNT(*) FROM web_sessions WHERE started_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND started_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR GROUP BY 1 ORDER BY 1",
                f"SELECT CAST(started_at_utc AS DATE) AS day, AVG(CASE WHEN converted_order_id IS NOT NULL THEN 1.0 ELSE 0.0 END) FROM web_sessions WHERE {p.between('CAST(started_at_utc AS DATE)')} GROUP BY 1 ORDER BY 1",
                ordered=True,
            )
        )
    return out


@template(
    "TR-F07-bkash-daily",
    "trend",
    "hard",
    "none",
    "payment_success",
    ["payments"],
    3,
    min_n=3,
    planted_fact="F07",
)
def tr_f07(ctx: Ctx, n: int) -> list[Instance]:
    qs = [
        (
            "Show the daily bKash payment success rate for August 2025 (Dhaka dates).",
            "2025-08-01",
            "2025-08-31",
            "",
        ),
        (
            "Were there days in August 2025 when bKash payment success dropped? Show the three worst days.",
            "2025-08-01",
            "2025-08-31",
            " ORDER BY success LIMIT 3",
        ),
        (
            "What was the bKash payment success rate on 12-14 August 2025 combined?",
            "2025-08-12",
            "2025-08-14",
            "agg",
        ),
    ]
    out = []
    for en, s, e, mode in qs[:n]:
        d = "CAST(created_at_utc + INTERVAL 6 HOUR AS DATE)"
        rate = "SUM(CASE WHEN status = 'captured' THEN 1 ELSE 0 END) * 1.0 / SUM(CASE WHEN status IN ('captured', 'failed') THEN 1 ELSE 0 END)"
        rate_alt = "COUNT(*) FILTER (WHERE status = 'captured') * 1.0 / COUNT(*) FILTER (WHERE status IN ('captured', 'failed'))"
        where = f"method = 'bkash' AND {d} BETWEEN DATE '{s}' AND DATE '{e}'"
        where_alt = f"method = 'bkash' AND created_at_utc >= TIMESTAMP '{s}' - INTERVAL 6 HOUR AND created_at_utc < TIMESTAMP '{e}' + INTERVAL 18 HOUR"
        if mode == "agg":
            g = f"SELECT {rate} AS success FROM payments WHERE {where}"
            a = f"SELECT {rate_alt} FROM payments WHERE {where_alt}"
        else:
            g = f"SELECT {d} AS day, {rate} AS success FROM payments WHERE {where} GROUP BY 1" + (
                mode or " ORDER BY 1"
            )
            a = (
                f"SELECT CAST(created_at_utc + INTERVAL 6 HOUR AS DATE) AS dd, {rate_alt} AS success FROM payments WHERE {where_alt} GROUP BY dd"
                + (mode or " ORDER BY 1")
            )
        out.append(
            I(
                en[:30],
                en,
                "২০২৫ সালের আগস্টে বিকাশ পেমেন্ট সফলতার হার দৈনিক কেমন ছিল?",
                "2025 er August e bKash payment success rate protidin kemon chilo?",
                g,
                a,
                ordered=True,
                planted_fact="F07",
            )
        )
    return out


@template(
    "TR-quarterly-net-revenue", "trend", "hard", "payment_retries", "net_revenue", ["orders", "payments"], 8
)
def tr_netrev(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, YEARS + FISCAL, n):
        q = "date_trunc('quarter', o.order_at)"
        g = f"""WITH o AS (SELECT o.order_id, {q} AS qtr FROM orders o WHERE {VALID} AND {odate(p)}),
                cap AS (SELECT o.qtr, SUM(p.amount_bdt) AS c FROM payments p JOIN o ON o.order_id = p.order_id WHERE p.status = 'captured' GROUP BY 1),
                ref AS (SELECT o.qtr, SUM(r.amount_bdt) AS rf FROM refunds r JOIN o ON o.order_id = r.order_id GROUP BY 1)
                SELECT cap.qtr AS quarter, cap.c - COALESCE(ref.rf, 0) AS net_revenue FROM cap LEFT JOIN ref ON ref.qtr = cap.qtr ORDER BY 1"""
        a = f"""SELECT date_trunc('quarter', o.order_at) AS qq, SUM((SELECT COALESCE(SUM(amount_bdt), 0) FROM payments p WHERE p.order_id = o.order_id AND p.status = 'captured')
                - (SELECT COALESCE(SUM(amount_bdt), 0) FROM refunds r WHERE r.order_id = o.order_id)) FROM orders o WHERE {VALID_ALT} AND {odate_alt(p)} GROUP BY 1 ORDER BY 1"""
        t = f"""WITH o AS (SELECT o.order_id, {q} AS qtr FROM orders o WHERE {VALID} AND {odate(p)}),
                cap AS (SELECT o.qtr, SUM(p.amount_bdt) AS c FROM payments p JOIN o ON o.order_id = p.order_id GROUP BY 1),
                ref AS (SELECT o.qtr, SUM(r.amount_bdt) AS rf FROM refunds r JOIN o ON o.order_id = r.order_id GROUP BY 1)
                SELECT cap.qtr AS quarter, cap.c - COALESCE(ref.rf, 0) AS net_revenue FROM cap LEFT JOIN ref ON ref.qtr = cap.qtr ORDER BY 1"""
        out.append(
            I(
                p.key,
                f"What was net revenue per quarter in {p.en}?",
                f"{p.bn} প্রতি প্রান্তিকে নিট রাজস্ব কত ছিল?",
                f"{p.bl} protita quarter e net revenue koto chilo?",
                g,
                a,
                t,
                ordered=True,
            )
        )
    return out


@template(
    "TR-monthly-delivery-days", "trend", "medium", "none", "avg_delivery_days", ["logistics", "orders"], 6
)
def tr_ddays(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, YEARS + FISCAL + QUARTERS[-4:], n):
        dd = "CAST(s.delivered_at_utc + INTERVAL 6 HOUR AS DATE)"
        sd = "CAST(s.shipped_at_utc + INTERVAL 6 HOUR AS DATE)"
        out.append(
            I(
                p.key,
                f"What was the average delivery time in days per month in {p.en}?",
                f"{p.bn} প্রতি মাসে গড় ডেলিভারি সময় কত দিন ছিল?",
                f"{p.bl} protimashe gore delivery te koto din legeche?",
                f"SELECT date_trunc('month', {dd}) AS month, AVG(date_diff('day', {sd}, {dd})) FROM shipments s JOIN orders o ON o.order_id = s.order_id WHERE s.status = 'delivered' AND NOT o.is_deleted AND {p.between(dd)} GROUP BY 1 ORDER BY 1",
                f"SELECT CAST(strftime({dd}, '%Y-%m-01') AS DATE), AVG({dd} - {sd}) FROM shipments s WHERE s.delivered_at_utc IS NOT NULL AND s.order_id IN (SELECT order_id FROM orders WHERE is_deleted = FALSE) AND {dd} >= DATE '{p.s}' AND {dd} <= DATE '{p.e}' GROUP BY 1 ORDER BY 1",
                ordered=True,
            )
        )
    return out


F05_GOLD = f"""SELECT CAST(o.order_at AS DATE) AS day, SUM({ORDER_VALUE}) AS gmv FROM orders o {FX_JOIN} WHERE {VALID} {{extra}}
               GROUP BY 1 ORDER BY gmv DESC LIMIT 1"""
F05_ALT = """SELECT d, g FROM (SELECT CAST(o.order_at AS DATE) AS d, SUM(CASE WHEN o.currency = 'BDT' THEN o.total_bdt ELSE o.total_amount * x.bdt_per_unit END) AS g
             FROM orders o LEFT JOIN currency_rates x ON x.rate_date = CAST(o.order_at AS DATE) AND x.currency = o.currency
             WHERE o.is_deleted = FALSE AND o.status IN ('delivered', 'in_transit', 'returned') {extra} GROUP BY 1) t QUALIFY RANK() OVER (ORDER BY g DESC) = 1"""


@template(
    "TR-F05-top-gmv-day",
    "trend",
    "hard",
    "none",
    "gmv",
    ["orders", "reference"],
    6,
    min_n=3,
    planted_fact="F05",
)
def tr_f05(ctx: Ctx, n: int) -> list[Instance]:
    cases = [
        ("Which single day had the highest GMV in the whole data period?", "", "F05"),
        ("What was our best day ever by GMV, and how much was it?", "", "F05"),
        ("Which day had the highest GMV in 2025?", f"AND {odate(YEARS[1])}", "F05"),
        ("Which day had the highest GMV in 2024?", f"AND {odate(YEARS[0])}", None),
        ("Which day had the highest GMV in 2026 so far?", f"AND {odate(YEARS[2])}", None),
        ("Which day in November 2025 had the highest GMV?", f"AND {odate(month(2025, 11))}", "F05"),
    ]
    out = []
    for en, extra, pf in cases[:n]:
        out.append(
            I(
                en[:40],
                en,
                "কোন দিনে জিএমভি সবচেয়ে বেশি ছিল?",
                "kon din e GMV sobcheye beshi chilo?",
                F05_GOLD.format(extra=extra),
                F05_ALT.format(extra=extra.replace("o.order_at", "o.order_at")),
                ordered=True,
                planted_fact=pf,
                must_mention=["2025-11-11"] if pf else [],
            )
        )
    return out


@template("TR-weekly-stockout", "trend", "medium", "none", "stockout_rate", ["inventory"], 6)
def tr_stock(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    pairs = [(w, q) for w in ctx.warehouses[:6] for q in QUARTERS[-4:]]
    for w, p in pick(ctx.rng, pairs, n):
        out.append(
            I(
                f"{w}:{p.key}",
                f"Show the weekly stockout rate at warehouse {w} in {p.en}.",
                f"{p.bn} {w} ওয়্যারহাউসে সাপ্তাহিক স্টক শেষ হওয়ার হার দেখাও।",
                f"{p.bl} {w} warehouse e weekly stockout rate dekhao.",
                f"SELECT s.snapshot_date, AVG(CASE WHEN s.on_hand_units = 0 THEN 1.0 ELSE 0.0 END) FROM inventory_snapshots s JOIN warehouses w ON w.warehouse_id = s.warehouse_id WHERE w.name = '{w}' AND {p.between('s.snapshot_date')} GROUP BY 1 ORDER BY 1",
                f"SELECT snapshot_date, COUNT(*) FILTER (WHERE on_hand_units = 0) * 1.0 / COUNT(*) FROM inventory_snapshots WHERE warehouse_id = (SELECT warehouse_id FROM warehouses WHERE name = '{w}') AND snapshot_date >= DATE '{p.s}' AND snapshot_date <= DATE '{p.e}' GROUP BY 1 ORDER BY 1",
                ordered=True,
            )
        )
    return out


# =============================================================================== comparison
@template("CP-app-vs-web", "comparison", "medium", "none", "gmv", ["orders", "reference"], 8)
def cp_channel(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        out.append(
            I(
                p.key,
                f"Compare GMV from the app and the website in {p.en}.",
                f"{p.bn} অ্যাপ ও ওয়েবসাইট থেকে আসা জিএমভি তুলনা করো।",
                f"{p.bl} app ar website er GMV compare koro.",
                f"SELECT o.channel, SUM({ORDER_VALUE}) AS gmv FROM orders o {FX_JOIN} WHERE {VALID} AND o.channel IN ('app', 'web') AND {odate(p)} GROUP BY 1",
                f"SELECT channel, SUM(CASE WHEN currency = 'BDT' THEN total_bdt ELSE total_amount * (SELECT bdt_per_unit FROM currency_rates x WHERE x.currency = 'USD' AND x.rate_date = CAST(o.order_at AS DATE)) END) FROM orders o WHERE {VALID_ALT} AND channel <> 'social_commerce' AND {odate_alt(p)} GROUP BY channel",
            )
        )
    return out


@template("CP-weekend-aov", "comparison", "medium", "none", "aov", ["orders", "reference"], 6)
def cp_weekend(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        out.append(
            I(
                p.key,
                f"Was the average order value higher on weekends or weekdays in {p.en}?",
                f"{p.bn} সপ্তাহান্তে নাকি কর্মদিবসে গড় অর্ডার মূল্য বেশি ছিল?",
                f"{p.bl} weekend e na weekday te average order value beshi chilo?",
                f"SELECT cd.is_weekend, AVG({ORDER_VALUE}) AS aov FROM orders o {FX_JOIN} JOIN calendar_dim cd ON cd.date = CAST(o.order_at AS DATE) WHERE {VALID} AND {odate(p)} GROUP BY 1",
                f"SELECT dayofweek(o.order_at) IN (5, 6) AS wk, SUM(CASE WHEN o.currency = 'BDT' THEN o.total_bdt ELSE o.total_amount * x.bdt_per_unit END) / COUNT(*) FROM orders o LEFT JOIN currency_rates x ON x.rate_date = CAST(o.order_at AS DATE) AND x.currency = o.currency WHERE {VALID_ALT} AND {odate_alt(p)} GROUP BY 1",
            )
        )
    return out


def _refund_rate_sql(p: Period, label: str) -> str:
    return (
        f"SELECT '{label}' AS period, AVG(CASE WHEN o.order_id IN (SELECT order_id FROM refunds) THEN 1.0 ELSE 0.0 END) AS refund_rate "
        f"FROM orders o WHERE NOT o.is_deleted AND {odate(p)} AND o.order_id IN (SELECT order_id FROM payments WHERE status = 'captured')"
    )


@template("CP-refund-months", "comparison", "medium", "none", "refund_rate", ["orders", "payments"], 6)
def cp_refund(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    pairs = [(MONTHS[i], MONTHS[i + 12]) for i in range(0, 21)]
    for a, b in pick(ctx.rng, pairs, n):
        g = f"{_refund_rate_sql(a, a.key)} UNION ALL {_refund_rate_sql(b, b.key)}"
        alt = (
            f"SELECT strftime(o.order_at, '%Y-%m') AS m, COUNT(DISTINCT r.order_id) * 1.0 / COUNT(DISTINCT o.order_id) FROM orders o JOIN payments p ON p.order_id = o.order_id AND p.status = 'captured' "
            f"LEFT JOIN refunds r ON r.order_id = o.order_id WHERE o.is_deleted = FALSE AND (({odate_alt(a)}) OR ({odate_alt(b)})) GROUP BY 1"
        )
        out.append(
            I(
                f"{a.key}:{b.key}",
                f"How did the refund rate in {b.en} compare with {a.en}?",
                f"{b.bn} রিফান্ডের হার {a.bn} তুলনায় কেমন ছিল?",
                f"{b.bl} refund rate {a.bl} er tulonay kemon chilo?",
                g,
                alt,
            )
        )
    return out


@template(
    "CP-cod-dhaka",
    "comparison",
    "hard",
    "payment_retries",
    "cod_share",
    ["payments", "orders", "reference"],
    8,
)
def cp_cod(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        grp = "CASE WHEN r.name = 'Dhaka' THEN 'Dhaka' ELSE 'Other divisions' END"
        out.append(
            I(
                p.key,
                f"Compare the cash-on-delivery share of orders in Dhaka division with all other divisions in {p.en}.",
                f"{p.bn} ঢাকা বিভাগ ও অন্যান্য বিভাগের ক্যাশ অন ডেলিভারি অর্ডারের অংশ তুলনা করো।",
                f"{p.bl} Dhaka division ar onno division er COD share compare koro.",
                f"SELECT {grp} AS area, AVG(CASE WHEN p.method = 'cod' THEN 1.0 ELSE 0.0 END) AS cod_share FROM orders o JOIN payments p ON p.order_id = o.order_id AND p.status = 'captured' {REGION_JOIN} WHERE NOT o.is_deleted AND {odate(p)} GROUP BY 1",
                f"SELECT CASE WHEN r.region_id = 1 THEN 'Dhaka' ELSE 'Other divisions' END, COUNT(*) FILTER (WHERE p.method = 'cod') * 1.0 / COUNT(*) FROM payments p JOIN orders o USING (order_id) JOIN customer_addresses a USING (address_id) JOIN cities c ON c.city_id = a.city_id JOIN regions r ON r.region_id = c.region_id WHERE p.status = 'captured' AND o.is_deleted = FALSE AND {odate_alt(p)} GROUP BY 1",
                f"SELECT {grp} AS area, AVG(CASE WHEN p.method = 'cod' THEN 1.0 ELSE 0.0 END) FROM orders o JOIN payments p ON p.order_id = o.order_id {REGION_JOIN} WHERE NOT o.is_deleted AND {odate(p)} GROUP BY 1",
            )
        )
    return out


@template("CP-local-vs-imported", "comparison", "medium", "none", "avg_rating", ["support", "catalog"], 6)
def cp_local(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, YEARS + FISCAL + QUARTERS[-3:], n):
        out.append(
            I(
                p.key,
                f"Compare the average rating of products from local brands and foreign brands in {p.en}.",
                f"{p.bn} দেশি ও বিদেশি ব্র্যান্ডের পণ্যের গড় রেটিং তুলনা করো।",
                f"{p.bl} deshi ar bideshi brand er product er average rating compare koro.",
                f"SELECT b.is_local, AVG(r.rating) FROM product_reviews r JOIN products p ON p.product_id = r.product_id JOIN brands b ON b.brand_id = p.brand_id WHERE NOT p.is_deleted AND {p.between('CAST(r.created_at AS DATE)')} GROUP BY 1",
                f"SELECT b.country = 'Bangladesh', SUM(r.rating) * 1.0 / COUNT(*) FROM product_reviews r JOIN products p USING (product_id) JOIN brands b USING (brand_id) WHERE p.is_deleted = FALSE AND r.created_at >= TIMESTAMP '{p.s}' AND r.created_at < TIMESTAMP '{p.e}' + INTERVAL 1 DAY GROUP BY 1",
            )
        )
    return out


@template("CP-tier-gmv", "comparison", "extra_hard", "scd2", "gmv", ["sellers", "orders"], 10)
def cp_tier(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        base = "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id"
        g = f"""SELECT th.tier, SUM(oi.line_total_bdt) AS gmv {base} JOIN seller_tier_history th ON th.seller_id = oi.seller_id
                AND th.valid_from <= CAST(o.order_at AS DATE) AND CAST(o.order_at AS DATE) < th.valid_to
                WHERE {VALID} AND {odate(p)} AND th.tier IN ('gold', 'silver') GROUP BY 1"""
        a = f"""SELECT t, SUM(v) FROM (SELECT (SELECT th.tier FROM seller_tier_history th WHERE th.seller_id = oi.seller_id AND CAST(o.order_at AS DATE) >= th.valid_from
                AND CAST(o.order_at AS DATE) < th.valid_to) AS t, oi.line_total_bdt AS v {base} WHERE {VALID_ALT} AND {odate_alt(p)}) x WHERE t IN ('gold', 'silver') GROUP BY t"""
        tr = f"""SELECT th.tier, SUM(oi.line_total_bdt) {base} JOIN seller_tier_history th ON th.seller_id = oi.seller_id AND th.valid_to = DATE '9999-12-31'
                 WHERE {VALID} AND {odate(p)} AND th.tier IN ('gold', 'silver') GROUP BY 1"""
        out.append(
            I(
                p.key,
                f"In {p.en}, did gold-tier sellers generate more GMV than silver-tier sellers (tier at the time of the order)?",
                f"{p.bn} গোল্ড টিয়ারের বিক্রেতারা কি সিলভার টিয়ারের চেয়ে বেশি জিএমভি এনেছে (অর্ডারের সময়ের টিয়ার অনুযায়ী)?",
                f"{p.bl} gold tier seller ra ki silver tier er cheye beshi GMV enechhe (order er somoyer tier onujayi)?",
                g,
                a,
                tr,
            )
        )
    return out


@template(
    "CP-return-dept", "comparison", "extra_hard", "none", "return_rate", ["orders", "payments", "catalog"], 10
)
def cp_return(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        dept = "split_part(c.path, ' > ', 1)"
        out.append(
            I(
                p.key,
                f"Compare the item return rate of Fashion and Electronics in {p.en}.",
                f"{p.bn} ফ্যাশন ও ইলেকট্রনিক্সের পণ্য ফেরতের হার তুলনা করো।",
                f"{p.bl} Fashion ar Electronics er return rate compare koro.",
                f"""SELECT {dept} AS dept, COUNT(DISTINCT r.order_item_id) * 1.0 / COUNT(DISTINCT oi.order_item_id) AS return_rate FROM order_items oi JOIN orders o ON o.order_id = oi.order_id
                         JOIN product_variants v ON v.variant_id = oi.variant_id JOIN products p ON p.product_id = v.product_id JOIN categories c ON c.category_id = p.category_id
                         LEFT JOIN returns r ON r.order_item_id = oi.order_item_id WHERE NOT o.is_deleted AND o.status IN ('delivered', 'returned') AND {odate(p)}
                         AND {dept} IN ('Fashion', 'Electronics') GROUP BY 1""",
                f"""SELECT c1.name, AVG(CASE WHEN EXISTS (SELECT 1 FROM returns r WHERE r.order_item_id = oi.order_item_id) THEN 1.0 ELSE 0.0 END) FROM order_items oi
                         JOIN orders o USING (order_id) JOIN product_variants v USING (variant_id) JOIN products p USING (product_id) JOIN categories c3 ON c3.category_id = p.category_id
                         JOIN categories c2 ON c2.category_id = c3.parent_category_id JOIN categories c1 ON c1.category_id = c2.parent_category_id
                         WHERE o.is_deleted = FALSE AND o.status IN ('delivered', 'returned') AND {odate_alt(p)} AND c1.name IN ('Fashion', 'Electronics') GROUP BY 1""",
            )
        )
    return out


@template(
    "CP-promo-quarters", "comparison", "medium", "none", "promo_redemption_rate", ["marketing", "orders"], 6
)
def cp_promo(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    pairs = [(QUARTERS[i], QUARTERS[i + 4]) for i in range(0, 7)]
    for a, b in pick(ctx.rng, pairs, n):
        g = " UNION ALL ".join(
            f"SELECT '{q.key}' AS quarter, AVG(CASE WHEN o.promo_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS rate FROM orders o WHERE {VALID} AND {odate(q)}"
            for q in (a, b)
        )
        alt = f"SELECT strftime(o.order_at, '%Y') || 'Q' || quarter(o.order_at), COUNT(o.promo_id) * 1.0 / COUNT(*) FROM orders o WHERE {VALID_ALT} AND (({odate_alt(a)}) OR ({odate_alt(b)})) GROUP BY 1"
        out.append(
            I(
                f"{a.key}:{b.key}",
                f"Compare the promo code redemption rate in {a.en} and {b.en}.",
                f"{a.bn} ও {b.bn} প্রোমো কোড ব্যবহারের হার তুলনা করো।",
                f"{a.bl} ar {b.bl} promo use er rate compare koro.",
                g,
                alt,
            )
        )
    return out


@template(
    "CP-classic-panjabi",
    "comparison",
    "medium",
    "duplicate_names",
    None,
    ["catalog", "orders", "sellers"],
    5,
    planted_fact="F03",
)
def cp_dup(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, [YEARS[0], YEARS[1], YEARS[2], FISCAL[0], FISCAL[1]], n):
        base = f"""FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN product_variants v ON v.variant_id = oi.variant_id
                   JOIN products p ON p.product_id = v.product_id JOIN sellers s ON s.seller_id = p.seller_id WHERE p.name = 'Classic Panjabi' AND {VALID} AND {odate(p)}"""
        out.append(
            I(
                p.key,
                f"There are two products called 'Classic Panjabi'. Compare their GMV in {p.en}, showing each product's seller.",
                f"'Classic Panjabi' নামে দুটি পণ্য আছে। {p.bn} প্রতিটির বিক্রেতাসহ তাদের জিএমভি তুলনা করো।",
                f"'Classic Panjabi' name e duita product ache. {p.bl} seller soho tader GMV compare koro.",
                f"SELECT p.product_id, s.name, SUM(oi.line_total_bdt) AS gmv {base} GROUP BY p.product_id, s.name",
                f"""SELECT v.product_id, (SELECT s.name FROM products x JOIN sellers s USING (seller_id) WHERE x.product_id = v.product_id), SUM(oi.line_total_bdt)
                         FROM order_items oi JOIN product_variants v USING (variant_id) WHERE v.product_id IN (SELECT product_id FROM products WHERE name = 'Classic Panjabi')
                         AND oi.order_id IN (SELECT order_id FROM orders o WHERE {VALID_ALT} AND {odate_alt(p)}) GROUP BY v.product_id""",
                f"SELECT p.name, MIN(s.name), SUM(oi.line_total_bdt) {base} GROUP BY p.name",
                planted_fact="F03",
            )
        )
    return out


@template("CP-gazipur-stockout", "comparison", "medium", "none", "stockout_rate", ["inventory"], 5)
def cp_gazipur(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, [YEARS[0], YEARS[1], YEARS[2], quarter(2026, 1), quarter(2025, 3)], n):
        out.append(
            I(
                p.key,
                f"Compare the stockout rate of Gazipur-1 and Gazipur-2 in {p.en}.",
                f"{p.bn} গাজীপুর-১ ও গাজীপুর-২ এর স্টক শেষ হওয়ার হার তুলনা করো।",
                f"{p.bl} Gazipur-1 ar Gazipur-2 er stockout rate compare koro.",
                f"SELECT w.name, AVG(CASE WHEN s.on_hand_units = 0 THEN 1.0 ELSE 0.0 END) FROM inventory_snapshots s JOIN warehouses w ON w.warehouse_id = s.warehouse_id WHERE w.name IN ('Gazipur-1', 'Gazipur-2') AND {p.between('s.snapshot_date')} GROUP BY 1",
                f"SELECT (SELECT name FROM warehouses w WHERE w.warehouse_id = s.warehouse_id), COUNT(*) FILTER (WHERE on_hand_units = 0) * 1.0 / COUNT(*) FROM inventory_snapshots s WHERE s.warehouse_id IN (SELECT warehouse_id FROM warehouses WHERE name LIKE 'Gazipur-%') AND s.snapshot_date >= DATE '{p.s}' AND s.snapshot_date <= DATE '{p.e}' GROUP BY s.warehouse_id",
                planted_fact="F08" if p.key == "2026" else None,
            )
        )
    return out


# =============================================================================== ratio (beyond single metrics)
@template("RT-gift-share", "ratio", "easy", "none", None, ["orders"], 6)
def rt_gift(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        out.append(
            I(
                p.key,
                f"What share of valid orders in {p.en} contained at least one gift-wrapped item?",
                f"{p.bn} বৈধ অর্ডারের কত অংশে অন্তত একটি গিফট-র‍্যাপ করা পণ্য ছিল?",
                f"{p.bl} valid order er koto percent e kompokkhe ekta gift item chilo?",
                f"SELECT AVG(CASE WHEN EXISTS (SELECT 1 FROM order_items oi WHERE oi.order_id = o.order_id AND oi.is_gift) THEN 1.0 ELSE 0.0 END) FROM orders o WHERE {VALID} AND {odate(p)}",
                f"SELECT COUNT(DISTINCT CASE WHEN oi.is_gift THEN o.order_id END) * 1.0 / COUNT(DISTINCT o.order_id) FROM orders o JOIN order_items oi USING (order_id) WHERE {VALID_ALT} AND {odate_alt(p)}",
            )
        )
    return out


@template("RT-resolved-24h", "ratio", "medium", "nulls", None, ["support"], 8)
def rt_24h(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[6:] + QUARTERS, n):
        d = p.between("CAST(created_at_utc + INTERVAL 6 HOUR AS DATE)")
        out.append(
            I(
                p.key,
                f"What share of all support tickets created in {p.en} were resolved within 24 hours?",
                f"{p.bn} তৈরি হওয়া সব সাপোর্ট টিকিটের কত অংশ ২৪ ঘণ্টার মধ্যে সমাধান হয়েছে?",
                f"{p.bl} toiri howa sob support ticket er koto percent 24 ghontar moddhe solve hoyeche?",
                f"SELECT AVG(CASE WHEN resolved_at_utc IS NOT NULL AND resolved_at_utc - created_at_utc <= INTERVAL 24 HOUR THEN 1.0 ELSE 0.0 END) FROM support_tickets WHERE {d}",
                f"SELECT COUNT(*) FILTER (WHERE date_diff('second', created_at_utc, resolved_at_utc) <= 86400) * 1.0 / COUNT(*) FROM support_tickets WHERE created_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND created_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR",
                f"SELECT AVG(CASE WHEN resolved_at_utc - created_at_utc <= INTERVAL 24 HOUR THEN 1.0 ELSE 0.0 END) FROM support_tickets WHERE {d} AND resolved_at_utc IS NOT NULL",
            )
        )
    return out


@template("RT-warehouse-stockout", "ratio", "easy", "none", "stockout_rate", ["inventory"], 8)
def rt_wh_stock(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    pairs = [(w, p) for w in ctx.warehouses[:11] for p in YEARS + QUARTERS[-4:]]
    for w, p in pick(ctx.rng, pairs, n):
        out.append(
            I(
                f"{w}:{p.key}",
                f"What was the stockout rate at warehouse {w} in {p.en}?",
                f"{p.bn} {w} ওয়্যারহাউসে স্টক শেষ হওয়ার হার কত ছিল?",
                f"{p.bl} {w} warehouse e stockout rate koto chilo?",
                f"SELECT AVG(CASE WHEN s.on_hand_units = 0 THEN 1.0 ELSE 0.0 END) FROM inventory_snapshots s JOIN warehouses w ON w.warehouse_id = s.warehouse_id WHERE w.name = '{w}' AND {p.between('s.snapshot_date')}",
                f"SELECT COUNT(*) FILTER (WHERE on_hand_units = 0) * 1.0 / COUNT(*) FROM inventory_snapshots WHERE warehouse_id IN (SELECT warehouse_id FROM warehouses WHERE name = '{w}') AND snapshot_date >= DATE '{p.s}' AND snapshot_date <= DATE '{p.e}'",
            )
        )
    return out


# =============================================================================== cohort
F06_GOLD = """WITH v AS (SELECT o.customer_id, o.order_at FROM orders o WHERE NOT o.is_deleted AND o.status NOT IN ('cancelled', 'canceled', 'payment_failed')),
    f AS (SELECT customer_id, MIN(order_at) AS first_at FROM v GROUP BY 1),
    rep AS (SELECT f.customer_id, MAX(CASE WHEN v.order_at > f.first_at AND v.order_at <= f.first_at + INTERVAL 90 DAY THEN 1 ELSE 0 END) AS repeated FROM f JOIN v ON v.customer_id = f.customer_id GROUP BY 1)
    SELECT c.acquisition_source AS campaign, AVG(rep.repeated) AS repeat_rate_90d FROM customers c JOIN rep ON rep.customer_id = c.customer_id
    JOIN marketing_campaigns m ON m.name = c.acquisition_source WHERE NOT c.is_deleted GROUP BY 1 HAVING COUNT(*) >= {k} ORDER BY repeat_rate_90d DESC LIMIT 1"""
F06_ALT = """SELECT c.acquisition_source, AVG(CASE WHEN EXISTS (SELECT 1 FROM orders o2 WHERE o2.customer_id = c.customer_id AND o2.is_deleted = FALSE
    AND o2.status IN ('delivered', 'in_transit', 'returned') AND o2.order_at > fo.first_at AND o2.order_at <= fo.first_at + INTERVAL 90 DAY) THEN 1.0 ELSE 0.0 END) AS r
    FROM customers c JOIN (SELECT customer_id, MIN(order_at) AS first_at FROM orders WHERE is_deleted = FALSE AND status IN ('delivered', 'in_transit', 'returned') GROUP BY 1) fo
    ON fo.customer_id = c.customer_id WHERE c.is_deleted = FALSE AND c.acquisition_source IN (SELECT name FROM marketing_campaigns)
    GROUP BY 1 HAVING COUNT(*) >= {k} ORDER BY r DESC LIMIT 1"""


@template(
    "CH-F06-campaign-repeat",
    "cohort",
    "extra_hard",
    "none",
    None,
    ["customers", "marketing", "orders"],
    3,
    min_n=3,
    planted_fact="F06",
)
def ch_f06(ctx: Ctx, n: int) -> list[Instance]:
    qs = [
        (
            "Customers acquired by which campaign have the best 90-day repeat purchase rate (campaigns with at least 20 acquired customers)?",
            20,
        ),
        (
            "Which acquisition campaign brought customers who most often ordered again within 90 days of their first order? Only consider campaigns with 20+ customers.",
            20,
        ),
        (
            "Among campaigns that acquired at least 10 customers, which has the highest 90-day repeat rate?",
            10,
        ),
    ]
    return [
        I(
            f"{k}:{i}",
            en,
            f"কোন ক্যাম্পেইনে আসা গ্রাহকদের ৯০ দিনের মধ্যে পুনরায় কেনার হার সবচেয়ে বেশি (কমপক্ষে {bn(k)} জন গ্রাহক)?",
            f"kon campaign theke asha customer der 90 din er moddhe abar kenar rate sobcheye beshi (kompokkhe {k} jon customer)?",
            F06_GOLD.format(k=k),
            F06_ALT.format(k=k),
            ordered=True,
            planted_fact="F06",
            must_mention=["Boishakh25"],
        )
        for i, (en, k) in enumerate(qs[:n])
    ]


@template("CH-signup-ordered", "cohort", "hard", "none", None, ["customers", "orders"], 10)
def ch_signup(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[:24], n):
        out.append(
            I(
                p.key,
                f"Of the customers who signed up in {p.en}, what share have placed at least one valid order?",
                f"{p.bn} নিবন্ধন করা গ্রাহকদের কত অংশ অন্তত একটি বৈধ অর্ডার দিয়েছে?",
                f"{p.bl} signup kora customer der koto percent kompokkhe ekta valid order diyeche?",
                f"SELECT AVG(CASE WHEN EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id AND {VALID}) THEN 1.0 ELSE 0.0 END) FROM customers c WHERE NOT c.is_deleted AND c.signup_month = DATE '{p.s}'",
                f"SELECT COUNT(DISTINCT o.customer_id) * 1.0 / (SELECT COUNT(*) FROM customers WHERE is_deleted = FALSE AND signup_month = DATE '{p.s}') FROM orders o JOIN customers c USING (customer_id) WHERE c.is_deleted = FALSE AND c.signup_month = DATE '{p.s}' AND {VALID_ALT}",
            )
        )
    return out


@template("CH-next-quarter-retention", "cohort", "extra_hard", "none", None, ["customers", "orders"], 10)
def ch_retention(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for i in pick(ctx.rng, list(range(len(QUARTERS) - 1)), n):
        q, q2 = QUARTERS[i], QUARTERS[i + 1]
        g = f"""WITH v AS (SELECT o.customer_id, o.order_at FROM orders o WHERE {VALID}), f AS (SELECT customer_id, MIN(order_at) AS first_at FROM v GROUP BY 1)
                SELECT COUNT(*) AS cohort, SUM(CASE WHEN EXISTS (SELECT 1 FROM v WHERE v.customer_id = f.customer_id AND {q2.between("CAST(v.order_at AS DATE)")}) THEN 1 ELSE 0 END) AS retained
                FROM f WHERE {q.between("CAST(f.first_at AS DATE)")}"""
        a = f"""SELECT COUNT(DISTINCT f.customer_id), COUNT(DISTINCT o.customer_id) FROM (SELECT customer_id FROM orders o WHERE {VALID_ALT} GROUP BY customer_id
                HAVING MIN(order_at) >= TIMESTAMP '{q.s}' AND MIN(order_at) < TIMESTAMP '{q.e}' + INTERVAL 1 DAY) f LEFT JOIN orders o ON o.customer_id = f.customer_id
                AND o.is_deleted = FALSE AND o.status IN ('delivered', 'in_transit', 'returned') AND {odate_alt(q2)}"""
        out.append(
            I(
                q.key,
                f"How many customers placed their first valid order in {q.en}, and how many of them ordered again in {q2.en}?",
                f"{q.bn} কতজন গ্রাহক প্রথম বৈধ অর্ডার দিয়েছে, এবং তাদের কতজন {q2.bn} আবার অর্ডার দিয়েছে?",
                f"{q.bl} koto jon customer prothom valid order diyeche, ar tader koto jon {q2.bl} abar order diyeche?",
                g,
                a,
            )
        )
    return out


@template("CH-gmv-per-cohort", "cohort", "extra_hard", "none", "gmv", ["customers", "orders", "reference"], 9)
def ch_cohort_gmv(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, [YEARS[1], YEARS[2], FISCAL[1]] + QUARTERS[-6:], n):
        out.append(
            I(
                p.key,
                f"In {p.en}, what was the average GMV per active customer for each signup-year cohort?",
                f"{p.bn} প্রতিটি নিবন্ধন-বছর গোষ্ঠীর সক্রিয় গ্রাহক প্রতি গড় জিএমভি কত?",
                f"{p.bl} protita signup year cohort er active customer prati gore GMV koto?",
                f"SELECT year(c.signup_month) AS cohort, SUM({ORDER_VALUE}) / COUNT(DISTINCT o.customer_id) FROM orders o {FX_JOIN} JOIN customers c ON c.customer_id = o.customer_id WHERE {VALID} AND {odate(p)} GROUP BY 1",
                f"SELECT cy, SUM(v) / COUNT(DISTINCT cid) FROM (SELECT CAST(strftime(c.signup_month, '%Y') AS INTEGER) AS cy, o.customer_id AS cid, CASE WHEN o.currency = 'BDT' THEN o.total_bdt ELSE o.total_amount * x.bdt_per_unit END AS v FROM orders o JOIN customers c USING (customer_id) LEFT JOIN currency_rates x ON x.rate_date = CAST(o.order_at AS DATE) AND x.currency = o.currency WHERE {VALID_ALT} AND {odate_alt(p)}) t GROUP BY cy",
            )
        )
    return out


@template("CH-vip-orders", "cohort", "extra_hard", "scd2", None, ["customers", "orders"], 10)
def ch_vip(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, [YEARS[0], YEARS[1], YEARS[2], FISCAL[0], FISCAL[1]] + QUARTERS[-5:], n):
        g = f"""SELECT COUNT(*) * 1.0 / COUNT(DISTINCT o.customer_id) FROM orders o JOIN customer_segment_history h ON h.customer_id = o.customer_id
                AND h.valid_from <= CAST(o.order_at AS DATE) AND CAST(o.order_at AS DATE) < h.valid_to WHERE {VALID} AND {odate(p)} AND h.segment = 'vip'"""
        a = f"""SELECT AVG(n) FROM (SELECT o.customer_id, COUNT(*) AS n FROM orders o WHERE {VALID_ALT} AND {odate_alt(p)} AND EXISTS (SELECT 1 FROM customer_segment_history h
                WHERE h.customer_id = o.customer_id AND h.segment = 'vip' AND CAST(o.order_at AS DATE) >= h.valid_from AND CAST(o.order_at AS DATE) < h.valid_to) GROUP BY 1)"""
        t = f"""SELECT COUNT(*) * 1.0 / COUNT(DISTINCT o.customer_id) FROM orders o JOIN customer_segment_history h ON h.customer_id = o.customer_id AND h.is_current
                WHERE {VALID} AND {odate(p)} AND h.segment = 'vip'"""
        out.append(
            I(
                p.key,
                f"In {p.en}, how many valid orders on average did customers place while they were in the 'vip' segment (segment at the time of each order)?",
                f"{p.bn} 'vip' সেগমেন্টে থাকাকালীন গ্রাহকরা গড়ে কতটি বৈধ অর্ডার দিয়েছে (প্রতিটি অর্ডারের সময়ের সেগমেন্ট অনুযায়ী)?",
                f"{p.bl} 'vip' segment e thakar somoy customer ra gore koyta valid order diyeche (order er somoyer segment onujayi)?",
                g,
                a,
                t,
            )
        )
    return out


@template("CH-loyalty-redeemers", "cohort", "medium", "none", None, ["customers"], 5)
def ch_loyalty(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, [YEARS[0], YEARS[1], YEARS[2], FISCAL[0], FISCAL[1]], n):
        out.append(
            I(
                p.key,
                f"What share of loyalty accounts redeemed points at least once in {p.en}?",
                f"{p.bn} কত শতাংশ লয়্যালটি অ্যাকাউন্ট অন্তত একবার পয়েন্ট রিডিম করেছে?",
                f"{p.bl} koto percent loyalty account kompokkhe ekbar point redeem koreche?",
                f"SELECT COUNT(DISTINCT CASE WHEN lt.txn_type = 'redeem' AND {p.between('CAST(lt.txn_at AS DATE)')} THEN la.loyalty_id END) * 1.0 / COUNT(DISTINCT la.loyalty_id) FROM loyalty_accounts la LEFT JOIN loyalty_transactions lt ON lt.loyalty_id = la.loyalty_id",
                f"SELECT AVG(CASE WHEN loyalty_id IN (SELECT loyalty_id FROM loyalty_transactions WHERE txn_type = 'redeem' AND txn_at >= TIMESTAMP '{p.s}' AND txn_at < TIMESTAMP '{p.e}' + INTERVAL 1 DAY) THEN 1.0 ELSE 0.0 END) FROM loyalty_accounts",
            )
        )
    return out


# =============================================================================== funnel
@template("FN-web-funnel", "funnel", "medium", "none", None, ["web"], 10)
def fn_web(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[6:] + QUARTERS, n):
        d = p.between("CAST(started_at_utc + INTERVAL 6 HOUR AS DATE)")
        out.append(
            I(
                p.key,
                f"Show the web funnel for {p.en}: sessions, sessions with add-to-cart, and converted sessions.",
                f"{p.bn} ওয়েব ফানেল দেখাও: সেশন, কার্টে যোগ করা সেশন এবং কনভার্টেড সেশন।",
                f"{p.bl} web funnel dekhao: session, add-to-cart session ar converted session.",
                f"SELECT COUNT(*) AS sessions, SUM(CASE WHEN added_to_cart THEN 1 ELSE 0 END) AS carts, COUNT(converted_order_id) AS converted FROM web_sessions WHERE {d}",
                f"SELECT COUNT(session_id), COUNT(*) FILTER (WHERE added_to_cart), COUNT(*) FILTER (WHERE converted_order_id IS NOT NULL) FROM web_sessions WHERE started_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND started_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR",
            )
        )
    return out


@template("FN-order-funnel", "funnel", "medium", "none", None, ["orders"], 10)
def fn_orders(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[3:-1], n):
        out.append(
            I(
                p.key,
                f"For orders placed in {p.en}, how many were placed, shipped, delivered and returned (excluding deleted orders)?",
                f"{p.bn} দেওয়া অর্ডারগুলোর মধ্যে কতগুলো দেওয়া, শিপ, ডেলিভারি এবং ফেরত হয়েছে (মুছে ফেলা অর্ডার বাদে)?",
                f"{p.bl} dewa order gulor moddhe koyta place, ship, deliver ar return hoyeche (deleted order bade)?",
                f"""SELECT COUNT(DISTINCT o.order_id) AS placed, COUNT(DISTINCT CASE WHEN e.status = 'shipped' THEN o.order_id END) AS shipped,
                         COUNT(DISTINCT CASE WHEN e.status = 'delivered' THEN o.order_id END) AS delivered, COUNT(DISTINCT CASE WHEN e.status = 'returned' THEN o.order_id END) AS returned
                         FROM orders o JOIN order_status_events e ON e.order_id = o.order_id WHERE NOT o.is_deleted AND {odate(p)}""",
                f"""SELECT COUNT(*), SUM(sh), SUM(dl), SUM(rt) FROM (SELECT o.order_id, MAX(CASE WHEN e.status = 'shipped' THEN 1 ELSE 0 END) AS sh,
                         MAX(CASE WHEN e.status = 'delivered' THEN 1 ELSE 0 END) AS dl, MAX(CASE WHEN e.status = 'returned' THEN 1 ELSE 0 END) AS rt
                         FROM orders o JOIN order_status_events e USING (order_id) WHERE o.is_deleted = FALSE AND {odate_alt(p)} GROUP BY o.order_id)""",
            )
        )
    return out


@template("FN-retry-share", "funnel", "hard", "none", None, ["payments", "orders"], 10)
def fn_retry(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[3:] + QUARTERS, n):
        out.append(
            I(
                p.key,
                f"Of orders placed in {p.en} that were eventually paid, what share had at least one failed payment attempt first?",
                f"{p.bn} যে অর্ডারগুলো শেষ পর্যন্ত পরিশোধ হয়েছে, তার কত অংশে আগে অন্তত একবার পেমেন্ট ব্যর্থ হয়েছিল?",
                f"{p.bl} je order gula sheshe pay hoyeche, tar koto percent e age kompokkhe ekbar payment fail hoyechilo?",
                f"""SELECT AVG(CASE WHEN EXISTS (SELECT 1 FROM payments f WHERE f.order_id = o.order_id AND f.status = 'failed') THEN 1.0 ELSE 0.0 END) FROM orders o
                         WHERE NOT o.is_deleted AND {odate(p)} AND EXISTS (SELECT 1 FROM payments c WHERE c.order_id = o.order_id AND c.status = 'captured')""",
                f"""SELECT COUNT(DISTINCT CASE WHEN p.status = 'failed' THEN p.order_id END) * 1.0 / COUNT(DISTINCT CASE WHEN p.status = 'captured' THEN p.order_id END)
                         FROM payments p JOIN orders o USING (order_id) WHERE o.is_deleted = FALSE AND {odate_alt(p)} AND p.order_id IN (SELECT order_id FROM payments WHERE status = 'captured')""",
            )
        )
    return out


@template("FN-review-share", "funnel", "hard", "none", None, ["support", "orders"], 10)
def fn_review(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[3:-2] + QUARTERS[:-1], n):
        out.append(
            I(
                p.key,
                f"Of delivered orders placed in {p.en}, what share received at least one product review?",
                f"{p.bn} ডেলিভারি হওয়া অর্ডারের কত অংশে অন্তত একটি রিভিউ এসেছে?",
                f"{p.bl} deliver howa order er koto percent e kompokkhe ekta review esheche?",
                f"SELECT AVG(CASE WHEN o.order_id IN (SELECT order_id FROM product_reviews) THEN 1.0 ELSE 0.0 END) FROM orders o WHERE NOT o.is_deleted AND o.status = 'delivered' AND {odate(p)}",
                f"SELECT COUNT(DISTINCT r.order_id) * 1.0 / COUNT(DISTINCT o.order_id) FROM orders o LEFT JOIN product_reviews r USING (order_id) WHERE o.is_deleted = FALSE AND o.status = 'delivered' AND {odate_alt(p)}",
            )
        )
    return out


@template("FN-campaign-traffic", "funnel", "medium", "none", None, ["web", "marketing"], 10)
def fn_campaign(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        d = p.between("CAST(ws.started_at_utc + INTERVAL 6 HOUR AS DATE)")
        out.append(
            I(
                p.key,
                f"For web sessions attributed to a marketing campaign in {p.en}, how many added to cart and how many converted, by campaign channel?",
                f"{p.bn} মার্কেটিং ক্যাম্পেইন থেকে আসা ওয়েব সেশনের মধ্যে কতগুলো কার্টে যোগ করেছে এবং কতগুলো কনভার্ট হয়েছে, ক্যাম্পেইন চ্যানেল অনুযায়ী?",
                f"{p.bl} campaign theke asha web session er moddhe koyta cart e add koreche ar koyta convert hoyeche, campaign channel onujayi?",
                f"SELECT m.channel, COUNT(*) AS sessions, SUM(CASE WHEN ws.added_to_cart THEN 1 ELSE 0 END) AS carts, COUNT(ws.converted_order_id) AS converted FROM web_sessions ws JOIN marketing_campaigns m ON m.campaign_id = ws.campaign_id WHERE {d} GROUP BY 1",
                f"SELECT (SELECT channel FROM marketing_campaigns m WHERE m.campaign_id = ws.campaign_id) AS ch, COUNT(session_id), COUNT(*) FILTER (WHERE added_to_cart), COUNT(*) FILTER (WHERE converted_order_id IS NOT NULL) FROM web_sessions ws WHERE ws.campaign_id IS NOT NULL AND ws.started_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND ws.started_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR GROUP BY 1",
            )
        )
    return out


@template(
    "TR-mom-growth-division", "trend", "extra_hard", "none", "gmv", ["orders", "customers", "reference"], 10
)
def tr_mom(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, [YEARS[0], YEARS[1], YEARS[2], FISCAL[0], FISCAL[1]] + QUARTERS[-6:], n):
        g = f"""WITH m AS (SELECT r.name AS division, date_trunc('month', o.order_at) AS month, SUM({ORDER_VALUE}) AS gmv FROM orders o {FX_JOIN} {REGION_JOIN}
                WHERE {VALID} AND {odate(p)} GROUP BY 1, 2)
                SELECT division, month, gmv, gmv / LAG(gmv) OVER (PARTITION BY division ORDER BY month) - 1 AS mom_growth FROM m ORDER BY division, month"""
        a = f"""SELECT d, m, g, g / prev - 1 FROM (SELECT d, m, g, LAG(g, 1) OVER (PARTITION BY d ORDER BY m) AS prev FROM (
                SELECT rg.name AS d, CAST(strftime(o.order_at, '%Y-%m-01') AS DATE) AS m, SUM(CASE WHEN o.currency = 'BDT' THEN o.total_bdt ELSE o.total_amount * x.bdt_per_unit END) AS g
                FROM orders o LEFT JOIN currency_rates x ON x.rate_date = CAST(o.order_at AS DATE) AND x.currency = o.currency JOIN customer_addresses ad USING (address_id)
                JOIN cities c ON c.city_id = ad.city_id JOIN regions rg ON rg.region_id = c.region_id WHERE {VALID_ALT} AND {odate_alt(p)} GROUP BY 1, 2) t) u ORDER BY d, m"""
        out.append(
            I(
                p.key,
                f"Show month-over-month GMV growth for each division in {p.en}.",
                f"{p.bn} প্রতিটি বিভাগের মাসওয়ারি জিএমভি প্রবৃদ্ধি দেখাও।",
                f"{p.bl} protita division er month-over-month GMV growth dekhao.",
                g,
                a,
                ordered=True,
            )
        )
    return out


# =============================================================================== extra non-metric templates
@template("AG-shipping-fees", "aggregation", "easy", "none", None, ["logistics"], 8)
def ag_fees(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[6:] + QUARTERS, n):
        d = p.between("CAST(shipped_at_utc + INTERVAL 6 HOUR AS DATE)")
        out.append(
            I(
                p.key,
                f"How much did we charge in shipping fees for shipments sent in {p.en} (Dhaka dates)?",
                f"{p.bn} (ঢাকার তারিখে) পাঠানো শিপমেন্টে মোট কত শিপিং ফি নেওয়া হয়েছে?",
                f"{p.bl} (Dhaka date) pathano shipment e mot koto shipping fee neya hoyeche?",
                f"SELECT SUM(shipping_fee_bdt) FROM shipments WHERE {d}",
                f"SELECT SUM(s.shipping_fee_bdt) FROM shipments s WHERE s.shipped_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND s.shipped_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR",
            )
        )
    return out


@template("AG-tickets-by-category", "aggregation", "easy", "none", None, ["support"], 8)
def ag_tickets(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[6:] + QUARTERS, n):
        d = p.between("CAST(created_at_utc + INTERVAL 6 HOUR AS DATE)")
        out.append(
            I(
                p.key,
                f"How many support tickets were opened in each category in {p.en}?",
                f"{p.bn} প্রতিটি ক্যাটাগরিতে কতগুলো সাপোর্ট টিকিট খোলা হয়েছে?",
                f"{p.bl} protita category te koyta support ticket khola hoyeche?",
                f"SELECT category, COUNT(*) FROM support_tickets WHERE {d} GROUP BY category",
                f"SELECT t.category, COUNT(t.ticket_id) FROM support_tickets t WHERE t.created_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND t.created_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR GROUP BY 1",
            )
        )
    return out


@template("RT-mobile-share", "ratio", "easy", "none", None, ["orders"], 8)
def rt_mobile(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[6:] + QUARTERS + YEARS, n):
        out.append(
            I(
                p.key,
                f"What share of valid orders in {p.en} came from Android or iOS devices?",
                f"{p.bn} বৈধ অর্ডারের কত অংশ অ্যান্ড্রয়েড বা আইওএস ডিভাইস থেকে এসেছে?",
                f"{p.bl} valid order er koto percent Android ba iOS theke esheche?",
                f"SELECT AVG(CASE WHEN o.device_type IN ('android', 'ios') THEN 1.0 ELSE 0.0 END) FROM orders o WHERE {VALID} AND {odate(p)}",
                f"SELECT COUNT(*) FILTER (WHERE o.device_type = 'android' OR o.device_type = 'ios') * 1.0 / COUNT(*) FROM orders o WHERE {VALID_ALT} AND {odate_alt(p)}",
            )
        )
    return out


@template("RT-usd-share", "ratio", "easy", "none", None, ["orders"], 6)
def rt_usd(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        out.append(
            I(
                p.key,
                f"What share of valid orders in {p.en} were paid in US dollars?",
                f"{p.bn} বৈধ অর্ডারের কত অংশ মার্কিন ডলারে ছিল?",
                f"{p.bl} valid order er koto percent USD te chilo?",
                f"SELECT AVG(CASE WHEN o.currency = 'USD' THEN 1.0 ELSE 0.0 END) FROM orders o WHERE {VALID} AND {odate(p)}",
                f"SELECT COUNT(*) FILTER (WHERE o.currency <> 'BDT') * 1.0 / COUNT(*) FROM orders o WHERE {VALID_ALT} AND {odate_alt(p)}",
            )
        )
    return out


@template("TR-monthly-tickets", "trend", "easy", "none", None, ["support"], 8)
def tr_tickets(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, YEARS + FISCAL + QUARTERS[-4:], n):
        d = "CAST(created_at_utc + INTERVAL 6 HOUR AS DATE)"
        out.append(
            I(
                p.key,
                f"How many support tickets were created each month in {p.en}?",
                f"{p.bn} প্রতি মাসে কতগুলো সাপোর্ট টিকিট তৈরি হয়েছে?",
                f"{p.bl} protimashe koyta support ticket toiri hoyeche?",
                f"SELECT date_trunc('month', {d}) AS month, COUNT(*) FROM support_tickets WHERE {p.between(d)} GROUP BY 1 ORDER BY 1",
                f"SELECT CAST(strftime(created_at_utc + INTERVAL 6 HOUR, '%Y-%m-01') AS DATE), COUNT(ticket_id) FROM support_tickets WHERE created_at_utc >= TIMESTAMP '{p.s}' - INTERVAL 6 HOUR AND created_at_utc < TIMESTAMP '{p.e}' + INTERVAL 18 HOUR GROUP BY 1 ORDER BY 1",
                ordered=True,
            )
        )
    return out


@template(
    "CP-dhaka-vs-ctg-orders", "comparison", "medium", "none", None, ["orders", "customers", "reference"], 8
)
def cp_div_orders(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, MONTHS[6:] + QUARTERS, n):
        out.append(
            I(
                p.key,
                f"Compare the number of valid orders from Dhaka division and Chattogram division in {p.en}.",
                f"{p.bn} ঢাকা বিভাগ ও চট্টগ্রাম বিভাগ থেকে আসা বৈধ অর্ডারের সংখ্যা তুলনা করো।",
                f"{p.bl} Dhaka division ar Chattogram division er valid order compare koro.",
                f"SELECT r.name, COUNT(*) FROM orders o {REGION_JOIN} WHERE {VALID} AND {odate(p)} AND r.name IN ('Dhaka', 'Chattogram') GROUP BY 1",
                f"SELECT rg.name, COUNT(o.order_id) FROM orders o JOIN customer_addresses a USING (address_id) JOIN cities c ON c.city_id = a.city_id JOIN regions rg ON rg.region_id = c.region_id WHERE {VALID_ALT} AND {odate_alt(p)} AND rg.region_id IN (1, 2) GROUP BY 1",
            )
        )
    return out


@template("RK-top-promo-codes", "ranking", "medium", "none", None, ["marketing", "orders"], 8)
def rk_promos(ctx: Ctx, n: int) -> list[Instance]:
    out = []
    for p in pick(ctx.rng, QUARTERS + YEARS, n):
        out.append(
            I(
                p.key,
                f"Which 5 promo codes were redeemed most often on valid orders in {p.en}?",
                f"{p.bn} বৈধ অর্ডারে কোন ৫টি প্রোমো কোড সবচেয়ে বেশি ব্যবহার হয়েছে?",
                f"{p.bl} valid order e kon 5 ta promo code sobcheye beshi use hoyeche?",
                f"SELECT pr.code, COUNT(*) AS uses FROM orders o JOIN promotions pr ON pr.promo_id = o.promo_id WHERE {VALID} AND {odate(p)} GROUP BY pr.code ORDER BY uses DESC, pr.code LIMIT 5",
                f"SELECT (SELECT code FROM promotions x WHERE x.promo_id = r.promo_id) AS c, COUNT(r.redemption_id) AS u FROM promotion_redemptions r JOIN orders o USING (order_id) WHERE {VALID_ALT} AND {odate_alt(p)} GROUP BY r.promo_id ORDER BY u DESC, c LIMIT 5",
                ordered=True,
            )
        )
    return out
