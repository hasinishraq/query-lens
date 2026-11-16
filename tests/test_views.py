import duckdb
import pytest

from bazaar_world.roles import ROLES, role_db_path
from bazaar_world.schema import TABLES

PII = {(t.name, c) for t in TABLES for c in t.pii}


def _cols(path):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return set(
            con.execute(
                "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='main'"
            ).fetchall()
        )
    finally:
        con.close()


@pytest.fixture(scope="module")
def role_cols(small_db):
    return {r: _cols(role_db_path(small_db.parent, "small", r)) for r in ROLES}


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


def test_role_db_blocks_base_tables_and_writes(small_db):
    con = duckdb.connect(
        str(role_db_path(small_db.parent, "small", "analyst")),
        read_only=True,
        config={"enable_external_access": False},
    )
    try:
        for sql in (
            "SELECT * FROM support_messages",
            "SELECT email FROM customers",
            f"ATTACH '{small_db.as_posix()}' AS b",
            "CREATE TABLE x AS SELECT 1",
            "DELETE FROM orders",
            "SELECT * FROM read_csv('world/params.yaml')",
        ):
            with pytest.raises(duckdb.Error):
                con.execute(sql)
    finally:
        con.close()
