"""Guard test suite (MySQL): every rejection rule, 30+ attack strings, allowed queries, output PII scan."""

import pymysql
import pytest

from querylens.config import SqlRuntime
from querylens.guard import (
    QueryTimeout,
    check_sql,
    cost_check,
    execute_with_timeout,
    open_explain_connection,
    open_role_connection,
    role_schema,
    scan_output,
)


@pytest.fixture(scope="module")
def schema(analyst_db):
    return role_schema("small", "analyst")


# (sql, expected rule) -- one or more per rule
RULE_CASES = [
    ("", "empty"),
    ("SELEC * FROM orders", "parse_error"),
    ("SELECT 1; SELECT 2", "multiple_statements"),
    ("DELETE FROM orders", "non_select"),
    ("UPDATE orders SET status = 'x'", "non_select"),
    ("INSERT INTO orders (order_id) VALUES (1)", "non_select"),
    ("REPLACE INTO orders (order_id) VALUES (1)", "non_select"),
    ("CREATE TABLE x AS SELECT 1", "non_select"),
    ("DROP TABLE orders", "non_select"),
    ("ALTER TABLE orders ADD COLUMN x INT", "non_select"),
    ("TRUNCATE TABLE orders", "non_select"),
    ("SET @x = 1", "non_select"),
    ("SET GLOBAL local_infile = 1", "non_select"),
    ("SHOW TABLES", "non_select"),
    ("USE bazaar_small", "non_select"),
    ("DESCRIBE orders", "non_select"),
    ("GRANT ALL ON *.* TO 'x'@'%'", "non_select"),
    ("SELECT * FROM orders INTO OUTFILE 'C:/out.csv'", "file_access"),
    ("SELECT body FROM product_reviews INTO DUMPFILE '/tmp/x'", "file_access"),
    ("SELECT LOAD_FILE('C:/Windows/win.ini')", "file_access"),
    ("SELECT SLEEP(30)", "file_access"),
    ("SELECT BENCHMARK(100000000, MD5('x'))", "file_access"),
    ("SELECT GET_LOCK('a', 10)", "file_access"),
    ("SELECT USER()", "file_access"),
    ("SELECT VERSION()", "file_access"),
    ("SELECT DATABASE()", "file_access"),
    ("SELECT @@version", "system_catalog"),
    ("SELECT @@datadir", "system_catalog"),
    ("SELECT * FROM bazaar_small.orders", "schema_reference"),
    ("SELECT * FROM bazaar_small_support.support_messages", "schema_reference"),
    ("SELECT * FROM information_schema.tables", "system_catalog"),
    ("SELECT * FROM mysql.user", "system_catalog"),
    ("SELECT * FROM performance_schema.threads", "system_catalog"),
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
    "SELECT * FROM (SELECT email FROM customers) t",
    "SELECT t.e FROM (SELECT email AS e FROM customers) t",
    "SELECT * FROM customers WHERE email LIKE '%@gmail.com'",
    "SELECT COUNT(*) FROM customers GROUP BY phone",
    "SELECT o.order_id FROM orders o JOIN customers c ON c.email = 'a@b.c'",
    "SELECT order_id FROM orders UNION ALL SELECT order_id FROM bazaar_small.orders",
    "SELECT body FROM support_messages WHERE ticket_id = 1",
    "SELECT * FROM bazaar_small_finance.seller_payouts",
    "SELECT * FROM `bazaar_small`.`customers`",
    "SELECT phone AS p FROM customers ORDER BY p",
    "SELECT birth_year FROM customers",
    "SELECT signup_at FROM customers",
    "SELECT o.order_id, (SELECT email FROM customers c WHERE c.customer_id = o.customer_id) FROM orders o",
    "SELECT CONCAT(LOAD_FILE('/etc/passwd'), '') AS x",
    "SELECT IF(1=1, SLEEP(20), 0)",
    "SELECT * FROM orders WHERE order_id = 1 AND SLEEP(5)",
    "SELECT @a := 1",
    "SELECT * FROM orders o, LATERAL (SELECT email FROM customers c WHERE c.customer_id = o.customer_id) x",
    "LOAD DATA LOCAL INFILE 'x.csv' INTO TABLE orders",
    "HANDLER orders OPEN",
    "CALL some_procedure()",
    "DO SLEEP(10)",
    "PREPARE s FROM 'SELECT email FROM customers'",
    "EXECUTE s",
    "LOCK TABLES orders READ",
    "KILL 1",
    "FLUSH PRIVILEGES",
    "SELECT * FROM sys.version",
    "SELECT CURRENT_USER()",
    "SELECT SYSTEM_USER()",
    "SELECT CONNECTION_ID()",
    "SELECT 1 INTO @x",
    "WITH c AS (SELECT email FROM customers) SELECT * FROM c",
    "SELECT * FROM orders INTO OUTFILE '/var/lib/mysql-files/x'",
]


@pytest.mark.parametrize("sql", ATTACKS)
def test_attacks_rejected(schema, sql):
    assert not check_sql(sql, schema).ok


ALLOWED = [
    "SELECT * FROM orders -- ; DELETE FROM orders",  # the DELETE is only a comment
    "SELECT COUNT(*) FROM orders",
    "SELECT status, COUNT(*) FROM orders GROUP BY status",
    "WITH v AS (SELECT * FROM orders WHERE NOT is_deleted) SELECT channel, COUNT(*) FROM v GROUP BY channel",
    "SELECT r.name, SUM(o.total_bdt) FROM orders o JOIN customer_addresses a ON a.address_id = o.address_id "
    "JOIN cities c ON c.city_id = a.city_id JOIN regions r ON r.region_id = c.region_id GROUP BY r.name",
    "SELECT * FROM orders LIMIT 5",
    "SELECT customer_id, signup_month FROM customers LIMIT 10",
    "SELECT order_id FROM orders UNION SELECT order_id FROM refunds",
    "SELECT DATE_FORMAT(order_at, '%Y-%m-01') AS m, COUNT(*) FROM orders GROUP BY m ORDER BY m",
    "SELECT CAST(created_at_utc + INTERVAL 6 HOUR AS DATE) d, COUNT(*) FROM payments GROUP BY d",
    "SELECT s.name, COUNT(*) FROM sellers s JOIN order_items oi USING (seller_id) GROUP BY s.name ORDER BY 2 DESC LIMIT 10",
    "SELECT DATEDIFF(CURDATE(), MAX(order_at)) FROM orders",
    "SELECT `status`, COUNT(*) FROM `orders` GROUP BY `status`",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed_queries(schema, sql):
    r = check_sql(sql, schema)
    assert r.ok, (r.rule, r.message)


def test_guard_reports_tables_and_columns(schema):
    r = check_sql(
        "SELECT o.status, SUM(oi.line_total_bdt) FROM orders o JOIN order_items oi USING (order_id) GROUP BY o.status",
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
    con = open_explain_connection("small", "analyst")
    try:
        assert cost_check(con, "SELECT * FROM order_items", rt).rule == "too_many_rows"
        assert cost_check(con, "SELECT * FROM order_items LIMIT 10", rt) is None
        assert cost_check(con, "SELECT seller_id, COUNT(*) FROM order_items GROUP BY seller_id", rt) is None
    finally:
        con.close()


def test_connection_is_read_only_and_isolated(analyst_db):
    con = open_role_connection("small", "analyst", SqlRuntime())
    try:
        for sql in (
            "CREATE TABLE z (a INT)",
            "SET GLOBAL max_connections = 10",
            "SELECT * FROM bazaar_small.customers",
        ):
            with pytest.raises(pymysql.MySQLError), con.cursor() as cur:
                cur.execute(sql)
    finally:
        con.close()


def test_execute_with_timeout_and_row_cap(analyst_db):
    con = open_role_connection("small", "analyst", SqlRuntime(timeout_s=1))
    try:
        cols, rows, total = execute_with_timeout(con, "SELECT order_id FROM orders", 1, 100)
        assert cols == ["order_id"]
        assert len(rows) == 100
        assert total > 100
        with pytest.raises(QueryTimeout):
            execute_with_timeout(
                con,
                "SELECT SUM(a.quantity * b.quantity) FROM order_items a JOIN order_items b ON a.seller_id <> b.seller_id",
                1,
                10,
            )
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
        ("The share was 0.017123456789 of orders", []),
    ],
)
def test_output_pii_scan(text, kinds):
    red, found = scan_output(text)
    assert found == kinds
    if kinds:
        assert "REDACTED" in red
    else:
        assert red == text
