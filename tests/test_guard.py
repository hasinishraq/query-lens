"""Guard test suite: every rejection rule, 30+ attack strings, allowed queries, output PII scan."""

import duckdb
import pytest

from querylens.config import SqlRuntime
from querylens.guard import (
    check_sql,
    cost_check,
    execute_with_timeout,
    open_role_connection,
    role_schema,
    scan_output,
)


@pytest.fixture(scope="module")
def schema(analyst_db):
    return role_schema(str(analyst_db), "analyst")


# (sql, expected rule) -- one or more per rule
RULE_CASES = [
    ("", "empty"),
    ("SELEC * FROM orders", "parse_error"),
    ("SELECT 1; SELECT 2", "multiple_statements"),
    ("DELETE FROM orders", "non_select"),
    ("UPDATE orders SET status = 'x'", "non_select"),
    ("INSERT INTO orders VALUES (1)", "non_select"),
    ("MERGE INTO orders USING orders o2 ON TRUE WHEN MATCHED THEN DELETE", "non_select"),
    ("CREATE TABLE x AS SELECT 1", "non_select"),
    ("DROP TABLE orders", "non_select"),
    ("ALTER TABLE orders ADD COLUMN x INT", "non_select"),
    ("ATTACH 'data/bazaar_small.duckdb' AS b", "non_select"),
    ("COPY orders TO 'out.csv'", "non_select"),
    ("PRAGMA show_tables", "non_select"),
    ("SET memory_limit = '10GB'", "non_select"),
    ("INSTALL httpfs", "non_select"),
    ("LOAD httpfs", "non_select"),
    ("CALL pragma_version()", "non_select"),
    ("DESCRIBE orders", "non_select"),
    ("SELECT * FROM read_csv('secrets.csv')", "file_access"),
    ("SELECT * FROM read_parquet('x.parquet')", "file_access"),
    ("SELECT * FROM read_json('x.json')", "file_access"),
    ("SELECT * FROM glob('*')", "file_access"),
    ("SELECT * FROM 'data/bazaar_small.duckdb'", "file_access"),
    ("SELECT * FROM 'https://evil.example/x.csv'", "file_access"),
    ("SELECT * FROM main.orders", "schema_reference"),
    ("SELECT * FROM base.main.customers", "schema_reference"),
    ("SELECT * FROM information_schema.tables", "system_catalog"),
    ("SELECT * FROM duckdb_tables()", "file_access"),
    ("SELECT getenv('HOME')", "file_access"),
    ("SELECT * FROM support_messages", "forbidden_table"),
    ("SELECT email FROM customers", "forbidden_column"),
    ("SELECT c.phone FROM customers c", "forbidden_column"),
    ("SELECT full_name FROM customers", "forbidden_column"),
    ("SELECT street_address FROM customer_addresses", "forbidden_column"),
    ("SELECT contact_email FROM sellers", "forbidden_column"),
    ("SELECT * FROM ordrs", "unknown_table"),
    ("SELECT revenue FROM orders", "unknown_column"),
]


@pytest.mark.parametrize(("sql", "rule"), RULE_CASES)
def test_rejection_rules(schema, sql, rule):
    r = check_sql(sql, schema)
    assert not r.ok
    assert r.rule == rule, r.message
    assert r.message  # actionable message


ATTACKS = [
    "SELECT * FROM orders; DROP TABLE orders",
    "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
    "SELECT * FROM (SELECT email FROM customers) t",
    "SELECT t.e FROM (SELECT email AS e FROM customers) t",
    "SELECT * FROM customers WHERE email LIKE '%@gmail.com'",
    "SELECT COUNT(*) FROM customers GROUP BY phone",
    "SELECT o.order_id FROM orders o JOIN customers c ON c.email = 'a@b.c'",
    "SELECT * FROM orders UNION ALL SELECT * FROM main.orders",
    "SELECT * FROM read_csv_auto('/etc/passwd')",
    "SELECT * FROM read_text('C:/Windows/win.ini')",
    "SELECT * FROM sqlite_scan('x.db', 't')",
    "SELECT * FROM postgres_scan('host=x', 'public', 't')",
    "SELECT current_setting('memory_limit')",
    "SELECT * FROM duckdb_settings()",
    "SELECT * FROM duckdb_secrets()",
    "SELECT * FROM pg_catalog.pg_tables",
    "SELECT * FROM temp.main.x",
    "SELECT body FROM support_messages WHERE ticket_id = 1",
    "SELECT * FROM v_support.support_messages",
    "SELECT * FROM v_finance.seller_payouts",
    "EXPORT DATABASE 'dump'",
    "CHECKPOINT",
    "VACUUM",
    "BEGIN TRANSACTION",
    "DETACH DATABASE base",
    "SELECT 1; ATTACH 'x.db'",
    "USE main",
    "SUMMARIZE customers",
    "SELECT * FROM query_table('customers')",
    "SELECT * FROM query('SELECT email FROM customers')",
    "SELECT phone AS p FROM customers ORDER BY p",
    "SELECT * FROM orders o, LATERAL (SELECT email FROM customers c WHERE c.customer_id = o.customer_id) x",
    "SELECT birth_year FROM customers",
    "SELECT signup_at FROM customers",
]


@pytest.mark.parametrize("sql", ATTACKS)
def test_attacks_rejected(schema, sql):
    assert not check_sql(sql, schema).ok


ALLOWED = [
    "SELECT * FROM orders -- ; DELETE FROM orders",  # the DELETE is only a comment
    "SELECT COUNT(*) FROM orders",
    "SELECT status, COUNT(*) FROM orders GROUP BY status",
    "WITH v AS (SELECT * FROM orders WHERE NOT is_deleted) SELECT channel, COUNT(*) FROM v GROUP BY 1",
    "SELECT r.name, SUM(o.total_bdt) FROM orders o JOIN customer_addresses a ON a.address_id = o.address_id "
    "JOIN cities c ON c.city_id = a.city_id JOIN regions r ON r.region_id = c.region_id GROUP BY 1",
    "SELECT * FROM orders LIMIT 5",
    "SELECT customer_id, signup_month FROM customers LIMIT 10",
    "SELECT order_id FROM orders UNION SELECT order_id FROM refunds",
    "SELECT date_trunc('month', order_at) AS m, COUNT(*) FROM orders GROUP BY ALL ORDER BY m",
    "SELECT CAST(created_at_utc + INTERVAL 6 HOUR AS DATE) d, COUNT(*) FROM payments GROUP BY 1",
    "SELECT s.name, COUNT(*) FROM sellers s JOIN order_items oi USING (seller_id) GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
    "FROM orders SELECT order_id LIMIT 3",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed_queries(schema, sql):
    r = check_sql(sql, schema)
    assert r.ok, (r.rule, r.message)


def test_guard_reports_tables_and_columns(schema):
    r = check_sql(
        "SELECT o.status, SUM(oi.line_total_bdt) FROM orders o JOIN order_items oi USING (order_id) "
        "GROUP BY 1",
        schema,
    )
    assert r.ok
    assert r.tables == ["order_items", "orders"]
    assert "order_items.line_total_bdt" in r.columns


def test_unknown_table_suggests(schema):
    r = check_sql("SELECT * FROM shipment", schema)
    assert "shipments" in r.message


def test_cost_check_rejects_unbounded(analyst_db):
    rt = SqlRuntime(explain_row_threshold=1000)
    con = open_role_connection(analyst_db, rt)
    try:
        assert cost_check(con, "SELECT * FROM order_items", rt).rule == "too_many_rows"
        assert cost_check(con, "SELECT * FROM order_items LIMIT 10", rt) is None
        assert cost_check(con, "SELECT seller_id, COUNT(*) FROM order_items GROUP BY 1", rt) is None
    finally:
        con.close()


def test_connection_is_read_only_and_isolated(analyst_db):
    con = open_role_connection(analyst_db, SqlRuntime())
    try:
        for sql in (
            "CREATE TABLE z AS SELECT 1",
            "SET memory_limit='8GB'",
            "SELECT * FROM read_csv('x.csv')",
        ):
            with pytest.raises(duckdb.Error):
                con.execute(sql)
    finally:
        con.close()


def test_execute_with_timeout_and_row_cap(analyst_db):
    con = open_role_connection(analyst_db, SqlRuntime())
    try:
        cols, rows, total = execute_with_timeout(con, "SELECT order_id FROM orders", 10, 100)
        assert cols == ["order_id"] and len(rows) == 100 and total > 100
        from querylens.guard import QueryTimeout

        with pytest.raises(QueryTimeout):
            execute_with_timeout(con, "SELECT COUNT(*) FROM range(1000000000) a, range(100000) b", 0.3, 10)
    finally:
        con.close()


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ("Call +8801712345678 now", ["phone"]),
        ("Number 01812345678 is listed", ["phone"]),
        ("mail rahim.ahmed12@gmail.com", ["email"]),
        ("Both a@b.co and +880 1912 345678", ["email", "phone"]),
        ("Net revenue was 1,234,567.89 BDT in 2026-Q2", []),
        ("Order 01234 and 2026-04-01", []),
    ],
)
def test_output_pii_scan(text, kinds):
    red, found = scan_output(text)
    assert found == kinds
    if kinds:
        assert "REDACTED" in red
    else:
        assert red == text
