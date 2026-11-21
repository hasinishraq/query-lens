from unittest.mock import MagicMock, patch

import pymysql
import pytest

from bazaar_world.mysqldb import ROLES, admin_connect, db_name, role_connect
from bazaar_world.roles import HOSTS, ensure_users, role_columns
from bazaar_world.schema import TABLES

PII = {(t.name, c) for t in TABLES for c in t.pii}


def test_role_users_are_created_for_remote_action_hosts():
    con = MagicMock()
    cursor = con.cursor.return_value.__enter__.return_value

    with patch("bazaar_world.roles.role_password", return_value="test-password"):
        ensure_users(con)

    assert "%" in HOSTS
    assert cursor.execute.call_count == len(ROLES) * len(HOSTS) * 2
    for call in cursor.execute.call_args_list:
        assert call.args[1][0].startswith("ql_")
        assert call.args[1][1] in HOSTS
        assert call.args[1][2] == "test-password"


@pytest.fixture(scope="module")
def role_cols(small_db):
    con = admin_connect()
    try:
        out = {}
        for r in ROLES:
            cols = role_columns(con, db_name("small", r))
            out[r] = {(t, c) for t, cs in cols.items() for c, _ in cs}
        return out
    finally:
        con.close()


def test_analyst_has_no_pii(role_cols):
    assert not (role_cols["analyst"] & PII)


def test_analyst_has_no_support_messages(role_cols):
    assert not any(t == "support_messages" for t, _ in role_cols["analyst"])


def test_analyst_signup_month_only(role_cols):
    cols = {c for t, c in role_cols["analyst"] if t == "customers"}
    assert "signup_month" in cols
    assert "signup_at" not in cols
    assert not {"gender", "birth_year"} & cols


def test_support_scope(role_cols):
    cols = role_cols["support"]
    assert ("customers", "full_name") in cols
    forbidden = {
        ("customers", "email"),
        ("customers", "phone"),
        ("customer_addresses", "street_address"),
        ("customer_addresses", "postcode"),
    }
    assert not (forbidden & cols)
    assert not any(t == "seller_payouts" for t, _ in cols)
    assert ("support_messages", "body") in cols


def test_finance_scope(role_cols):
    tables = {t for t, _ in role_cols["finance"]}
    assert {"payments", "refunds", "returns", "seller_payouts", "currency_rates", "orders"} <= tables
    assert "customers" not in tables
    assert "support_messages" not in tables
    assert not any(c == "customer_id" for t, c in role_cols["finance"] if t == "orders")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM support_messages",
        "SELECT email FROM customers",
        "SELECT COUNT(*) FROM bazaar_small.customers",
        "SELECT COUNT(*) FROM bazaar_small_support.support_messages",
        "CREATE TABLE x (a INT)",
        "DELETE FROM orders",
        "UPDATE orders SET status = 'x'",
        "SELECT 1 INTO OUTFILE 'C:/q.txt'",
        "DROP DATABASE bazaar_small_analyst",
    ],
)
def test_role_user_cannot(small_db, sql):
    """Physical enforcement: the analyst MySQL user cannot reach base tables, PII, other roles, or write."""
    con = role_connect("small", "analyst")
    try:
        with pytest.raises(pymysql.MySQLError), con.cursor() as cur:
            cur.execute(sql)
    finally:
        con.close()


def test_load_file_returns_nothing(small_db):
    con = role_connect("small", "analyst")
    try:
        with con.cursor() as cur:
            cur.execute("SELECT LOAD_FILE('C:/Windows/win.ini')")
            assert cur.fetchone()[0] is None  # no FILE privilege
    finally:
        con.close()
