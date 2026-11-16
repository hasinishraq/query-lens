"""BazaarDW physical schema: 35 tables, 266 columns (CLAUDE.md section 6.3).

Every table is listed with its columns in order and a logical type (mapped to MySQL by mysql_type()). The first column is the
primary key, except ``inventory_snapshots`` whose key is (snapshot_date, warehouse_id, variant_id).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Table:
    name: str
    domain: str
    columns: tuple[tuple[str, str], ...]
    pk: tuple[str, ...] = ()
    fixed: bool = False  # identical in both sizes
    full_rows: int = 0
    pii: tuple[str, ...] = field(default_factory=tuple)
    untrusted: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, ...]:
        return self.pk or (self.columns[0][0],)

    @property
    def column_names(self) -> list[str]:
        return [c for c, _ in self.columns]


def _t(name: str, domain: str, rows: int, cols: str, **kw: object) -> Table:
    parsed: list[tuple[str, str]] = []
    for line in cols.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        col, typ = line.split(None, 1)
        parsed.append((col, typ.strip()))
    return Table(name=name, domain=domain, columns=tuple(parsed), full_rows=rows, **kw)  # type: ignore[arg-type]


TABLES: list[Table] = [
    # ---------------- Reference ----------------
    _t(
        "regions",
        "reference",
        8,
        """
        region_id INTEGER
        name VARCHAR
        name_bn VARCHAR
        population_m DOUBLE
    """,
        fixed=True,
    ),
    _t(
        "cities",
        "reference",
        64,
        """
        city_id INTEGER
        region_id INTEGER
        name VARCHAR
        name_bn VARCHAR
        is_metro BOOLEAN
        courier_zone VARCHAR
    """,
        fixed=True,
    ),
    _t(
        "calendar_dim",
        "reference",
        1035,
        """
        date_key INTEGER
        date DATE
        year INTEGER
        quarter INTEGER
        month INTEGER
        month_name VARCHAR
        week_of_year INTEGER
        day_of_week VARCHAR
        is_weekend BOOLEAN
        is_holiday BOOLEAN
        holiday_name VARCHAR
        fiscal_year INTEGER
    """,
        fixed=True,
    ),
    _t(
        "currency_rates",
        "reference",
        1035,
        """
        rate_date DATE
        currency VARCHAR
        bdt_per_unit DOUBLE
        source VARCHAR
    """,
        fixed=True,
        pk=("rate_date", "currency"),
    ),
    # ---------------- Customers ----------------
    _t(
        "customers",
        "customers",
        10_000,
        """
        customer_id INTEGER
        full_name VARCHAR
        email VARCHAR
        phone VARCHAR
        gender VARCHAR
        birth_year INTEGER
        city_id INTEGER
        signup_at TIMESTAMP
        signup_channel VARCHAR
        acquisition_source VARCHAR
        is_verified BOOLEAN
        preferred_language VARCHAR
        is_deleted BOOLEAN
        updated_at TIMESTAMP
    """,
        pii=("full_name", "email", "phone"),
    ),
    _t(
        "customer_addresses",
        "customers",
        13_000,
        """
        address_id INTEGER
        customer_id INTEGER
        city_id INTEGER
        street_address VARCHAR
        postcode VARCHAR
        address_type VARCHAR
        is_default BOOLEAN
        created_at TIMESTAMP
        is_deleted BOOLEAN
    """,
        pii=("street_address", "postcode"),
    ),
    _t(
        "customer_segment_history",
        "customers",
        16_000,
        """
        segment_hist_id INTEGER
        customer_id INTEGER
        segment VARCHAR
        valid_from DATE
        valid_to DATE
        is_current BOOLEAN
    """,
    ),
    _t(
        "loyalty_accounts",
        "customers",
        4_500,
        """
        loyalty_id INTEGER
        customer_id INTEGER
        tier VARCHAR
        points_balance INTEGER
        enrolled_at TIMESTAMP
        status VARCHAR
    """,
    ),
    _t(
        "loyalty_transactions",
        "customers",
        36_000,
        """
        loyalty_txn_id INTEGER
        loyalty_id INTEGER
        order_id INTEGER
        txn_type VARCHAR
        points INTEGER
        txn_at TIMESTAMP
        expires_at TIMESTAMP
    """,
    ),
    # ---------------- Catalog ----------------
    _t(
        "categories",
        "catalog",
        300,
        """
        category_id INTEGER
        parent_category_id INTEGER
        name VARCHAR
        level INTEGER
        path VARCHAR
    """,
        fixed=True,
    ),
    _t(
        "brands",
        "catalog",
        400,
        """
        brand_id INTEGER
        name VARCHAR
        country VARCHAR
        is_local BOOLEAN
    """,
    ),
    _t(
        "products",
        "catalog",
        4_000,
        """
        product_id INTEGER
        seller_id INTEGER
        brand_id INTEGER
        category_id INTEGER
        name VARCHAR
        description VARCHAR
        list_price_bdt DOUBLE
        created_at TIMESTAMP
        status VARCHAR
        is_deleted BOOLEAN
        weight_g INTEGER
        is_imported BOOLEAN
    """,
    ),
    _t(
        "product_variants",
        "catalog",
        11_000,
        """
        variant_id INTEGER
        product_id INTEGER
        sku VARCHAR
        size VARCHAR
        colour VARCHAR
        price_delta_bdt DOUBLE
        barcode VARCHAR
        is_active BOOLEAN
    """,
    ),
    _t(
        "product_price_history",
        "catalog",
        30_000,
        """
        price_hist_id INTEGER
        variant_id INTEGER
        price_bdt DOUBLE
        valid_from DATE
        valid_to DATE
        change_reason VARCHAR
    """,
    ),
    # ---------------- Sellers ----------------
    _t(
        "sellers",
        "sellers",
        300,
        """
        seller_id INTEGER
        name VARCHAR
        legal_name VARCHAR
        city_id INTEGER
        joined_at TIMESTAMP
        seller_type VARCHAR
        contact_email VARCHAR
        contact_phone VARCHAR
        status VARCHAR
        rating_avg DOUBLE
    """,
        pii=("contact_email", "contact_phone"),
    ),
    _t(
        "seller_tier_history",
        "sellers",
        700,
        """
        tier_hist_id INTEGER
        seller_id INTEGER
        tier VARCHAR
        valid_from DATE
        valid_to DATE
    """,
    ),
    _t(
        "seller_payouts",
        "sellers",
        9_000,
        """
        payout_id INTEGER
        seller_id INTEGER
        period_start DATE
        period_end DATE
        gross_bdt DOUBLE
        commission_bdt DOUBLE
        paid_at TIMESTAMP
    """,
    ),
    # ---------------- Inventory ----------------
    _t(
        "warehouses",
        "inventory",
        12,
        """
        warehouse_id INTEGER
        name VARCHAR
        city_id INTEGER
        capacity_units INTEGER
        opened_at DATE
        is_active BOOLEAN
    """,
        fixed=True,
    ),
    _t(
        "inventory_snapshots",
        "inventory",
        40_000,
        """
        snapshot_date DATE
        warehouse_id INTEGER
        variant_id INTEGER
        on_hand_units INTEGER
        reserved_units INTEGER
        reorder_point INTEGER
    """,
        pk=("snapshot_date", "warehouse_id", "variant_id"),
    ),
    # ---------------- Orders ----------------
    _t(
        "orders",
        "orders",
        55_000,
        """
        order_id INTEGER
        customer_id INTEGER
        address_id INTEGER
        order_at TIMESTAMP
        status VARCHAR
        channel VARCHAR
        currency VARCHAR
        total_amount DOUBLE
        total_bdt DOUBLE
        promo_id INTEGER
        is_deleted BOOLEAN
        updated_at TIMESTAMP
        device_type VARCHAR
    """,
    ),
    _t(
        "order_items",
        "orders",
        140_000,
        """
        order_item_id INTEGER
        order_id INTEGER
        variant_id INTEGER
        seller_id INTEGER
        quantity INTEGER
        unit_price DOUBLE
        discount_bdt DOUBLE
        line_total_bdt DOUBLE
        is_gift BOOLEAN
    """,
    ),
    _t(
        "order_status_events",
        "orders",
        170_000,
        """
        event_id INTEGER
        order_id INTEGER
        status VARCHAR
        event_at_utc TIMESTAMP
        actor VARCHAR
    """,
    ),
    # ---------------- Payments ----------------
    _t(
        "payments",
        "payments",
        58_000,
        """
        payment_id INTEGER
        order_id INTEGER
        method VARCHAR
        provider VARCHAR
        attempt_no INTEGER
        status VARCHAR
        amount_bdt DOUBLE
        created_at_utc TIMESTAMP
        captured_at_utc TIMESTAMP
    """,
    ),
    _t(
        "refunds",
        "payments",
        3_300,
        """
        refund_id INTEGER
        order_id INTEGER
        payment_id INTEGER
        amount_bdt DOUBLE
        reason VARCHAR
        requested_at TIMESTAMP
        processed_at TIMESTAMP
    """,
    ),
    _t(
        "returns",
        "payments",
        2_600,
        """
        return_id INTEGER
        order_item_id INTEGER
        quantity INTEGER
        reason VARCHAR
        status VARCHAR
        requested_at TIMESTAMP
        received_at TIMESTAMP
        refund_id INTEGER
    """,
    ),
    # ---------------- Logistics ----------------
    _t(
        "couriers",
        "logistics",
        12,
        """
        courier_id INTEGER
        name VARCHAR
        service_level VARCHAR
        is_third_party BOOLEAN
    """,
        fixed=True,
    ),
    _t(
        "shipments",
        "logistics",
        56_000,
        """
        shipment_id INTEGER
        order_id INTEGER
        courier_id INTEGER
        warehouse_id INTEGER
        shipped_at_utc TIMESTAMP
        promised_date DATE
        delivered_at_utc TIMESTAMP
        status VARCHAR
        shipping_fee_bdt DOUBLE
        attempt_count INTEGER
    """,
    ),
    _t(
        "shipment_events",
        "logistics",
        150_000,
        """
        shipment_event_id INTEGER
        shipment_id INTEGER
        event_type VARCHAR
        event_at_utc TIMESTAMP
        city_id INTEGER
        note VARCHAR
    """,
        untrusted=("note",),
    ),
    # ---------------- Marketing ----------------
    _t(
        "marketing_campaigns",
        "marketing",
        120,
        """
        campaign_id INTEGER
        name VARCHAR
        channel VARCHAR
        start_date DATE
        end_date DATE
        budget_bdt DOUBLE
        objective VARCHAR
        owner_team VARCHAR
    """,
        fixed=True,
    ),
    _t(
        "promotions",
        "marketing",
        400,
        """
        promo_id INTEGER
        campaign_id INTEGER
        code VARCHAR
        promo_type VARCHAR
        discount_value DOUBLE
        min_order_bdt DOUBLE
        start_at TIMESTAMP
        end_at TIMESTAMP
        max_redemptions INTEGER
    """,
        fixed=True,
    ),
    _t(
        "promotion_redemptions",
        "marketing",
        15_000,
        """
        redemption_id INTEGER
        promo_id INTEGER
        order_id INTEGER
        customer_id INTEGER
        discount_bdt DOUBLE
        redeemed_at TIMESTAMP
    """,
    ),
    # ---------------- Web ----------------
    _t(
        "web_sessions",
        "web",
        120_000,
        """
        session_id INTEGER
        customer_id INTEGER
        started_at_utc TIMESTAMP
        ended_at_utc TIMESTAMP
        device_type VARCHAR
        traffic_source VARCHAR
        campaign_id INTEGER
        landing_page VARCHAR
        pages_viewed INTEGER
        added_to_cart BOOLEAN
        converted_order_id INTEGER
    """,
    ),
    # ---------------- Support and reviews ----------------
    _t(
        "support_tickets",
        "support",
        5_000,
        """
        ticket_id INTEGER
        customer_id INTEGER
        order_id INTEGER
        channel VARCHAR
        category VARCHAR
        priority VARCHAR
        status VARCHAR
        created_at_utc TIMESTAMP
        resolved_at_utc TIMESTAMP
        agent_id INTEGER
    """,
    ),
    _t(
        "support_messages",
        "support",
        25_000,
        """
        message_id INTEGER
        ticket_id INTEGER
        sender_type VARCHAR
        sent_at_utc TIMESTAMP
        body VARCHAR
        language VARCHAR
    """,
        untrusted=("body",),
    ),
    _t(
        "product_reviews",
        "support",
        18_000,
        """
        review_id INTEGER
        product_id INTEGER
        customer_id INTEGER
        order_id INTEGER
        rating INTEGER
        title VARCHAR
        body VARCHAR
        created_at TIMESTAMP
    """,
        untrusted=("title", "body"),
    ),
]

TABLE_BY_NAME: dict[str, Table] = {t.name: t for t in TABLES}

SIZES: dict[str, float] = {"small": 0.1, "full": 1.0}
SF_TO_SIZE: dict[str, str] = {"0.1": "small", "1": "full", "1.0": "full", "0.01": "small"}


def target_rows(table: str, size: str) -> int:
    t = TABLE_BY_NAME[table]
    if t.fixed:
        return t.full_rows
    return round(t.full_rows * SIZES[size])


def total_target(size: str) -> int:
    return sum(target_rows(t.name, size) for t in TABLES)


# Foreign keys: (table, column) -> (ref_table, ref_column). Nullable FKs are allowed to be NULL.
FOREIGN_KEYS: dict[tuple[str, str], tuple[str, str]] = {
    ("cities", "region_id"): ("regions", "region_id"),
    ("customers", "city_id"): ("cities", "city_id"),
    ("customer_addresses", "customer_id"): ("customers", "customer_id"),
    ("customer_addresses", "city_id"): ("cities", "city_id"),
    ("customer_segment_history", "customer_id"): ("customers", "customer_id"),
    ("loyalty_accounts", "customer_id"): ("customers", "customer_id"),
    ("loyalty_transactions", "loyalty_id"): ("loyalty_accounts", "loyalty_id"),
    ("loyalty_transactions", "order_id"): ("orders", "order_id"),
    ("categories", "parent_category_id"): ("categories", "category_id"),
    ("products", "seller_id"): ("sellers", "seller_id"),
    ("products", "brand_id"): ("brands", "brand_id"),
    ("products", "category_id"): ("categories", "category_id"),
    ("product_variants", "product_id"): ("products", "product_id"),
    ("product_price_history", "variant_id"): ("product_variants", "variant_id"),
    ("sellers", "city_id"): ("cities", "city_id"),
    ("seller_tier_history", "seller_id"): ("sellers", "seller_id"),
    ("seller_payouts", "seller_id"): ("sellers", "seller_id"),
    ("warehouses", "city_id"): ("cities", "city_id"),
    ("inventory_snapshots", "warehouse_id"): ("warehouses", "warehouse_id"),
    ("inventory_snapshots", "variant_id"): ("product_variants", "variant_id"),
    ("orders", "customer_id"): ("customers", "customer_id"),
    ("orders", "address_id"): ("customer_addresses", "address_id"),
    ("orders", "promo_id"): ("promotions", "promo_id"),
    ("order_items", "order_id"): ("orders", "order_id"),
    ("order_items", "variant_id"): ("product_variants", "variant_id"),
    ("order_items", "seller_id"): ("sellers", "seller_id"),
    ("order_status_events", "order_id"): ("orders", "order_id"),
    ("payments", "order_id"): ("orders", "order_id"),
    ("refunds", "order_id"): ("orders", "order_id"),
    ("refunds", "payment_id"): ("payments", "payment_id"),
    ("returns", "order_item_id"): ("order_items", "order_item_id"),
    ("returns", "refund_id"): ("refunds", "refund_id"),
    ("shipments", "order_id"): ("orders", "order_id"),
    ("shipments", "courier_id"): ("couriers", "courier_id"),
    ("shipments", "warehouse_id"): ("warehouses", "warehouse_id"),
    ("shipment_events", "shipment_id"): ("shipments", "shipment_id"),
    ("shipment_events", "city_id"): ("cities", "city_id"),
    ("promotions", "campaign_id"): ("marketing_campaigns", "campaign_id"),
    ("promotion_redemptions", "promo_id"): ("promotions", "promo_id"),
    ("promotion_redemptions", "order_id"): ("orders", "order_id"),
    ("promotion_redemptions", "customer_id"): ("customers", "customer_id"),
    ("web_sessions", "customer_id"): ("customers", "customer_id"),
    ("web_sessions", "campaign_id"): ("marketing_campaigns", "campaign_id"),
    ("web_sessions", "converted_order_id"): ("orders", "order_id"),
    ("support_tickets", "customer_id"): ("customers", "customer_id"),
    ("support_tickets", "order_id"): ("orders", "order_id"),
    ("support_messages", "ticket_id"): ("support_tickets", "ticket_id"),
    ("product_reviews", "product_id"): ("products", "product_id"),
    ("product_reviews", "customer_id"): ("customers", "customer_id"),
    ("product_reviews", "order_id"): ("orders", "order_id"),
}


# ---------------------------------------------------------------- MySQL physical types
LONG_TEXT = {"description", "body", "note", "title", "path", "street_address", "landing_page"}
_MYSQL = {"INTEGER": "INT", "DOUBLE": "DOUBLE", "BOOLEAN": "BOOLEAN", "TIMESTAMP": "DATETIME", "DATE": "DATE"}


def mysql_type(column: str, logical: str) -> str:
    """MySQL column type for a logical (schema) type."""
    if logical == "VARCHAR":
        return "TEXT" if column in LONG_TEXT else "VARCHAR(255)"
    return _MYSQL[logical]


def mysql_data_type(column: str, logical: str) -> str:
    """The DATA_TYPE that information_schema reports for :func:`mysql_type` (BOOLEAN is stored as tinyint)."""
    return {
        "INT": "int",
        "DOUBLE": "double",
        "BOOLEAN": "tinyint",
        "DATETIME": "datetime",
        "DATE": "date",
        "TEXT": "text",
        "VARCHAR(255)": "varchar",
    }[mysql_type(column, logical)]


# indexes that make the warehouse usable without full scans (FK columns and common date filters)
EXTRA_INDEXES: list[tuple[str, str]] = [
    ("orders", "order_at"),
    ("payments", "created_at_utc"),
    ("shipments", "delivered_at_utc"),
    ("shipments", "shipped_at_utc"),
    ("web_sessions", "started_at_utc"),
    ("support_tickets", "created_at_utc"),
    ("product_reviews", "created_at"),
    ("refunds", "requested_at"),
    ("loyalty_transactions", "txn_at"),
    ("inventory_snapshots", "snapshot_date"),
    ("customers", "signup_at"),
    ("products", "name"),
    ("sellers", "name"),
]
