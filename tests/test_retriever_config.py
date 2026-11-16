import pytest

from querylens.config import AgentConfig, load_config
from querylens.retriever import SchemaRetriever, recall_at_k, tokenize


@pytest.fixture(scope="module")
def retr():
    return SchemaRetriever("analyst", use_embeddings=False)


@pytest.mark.parametrize(
    ("q", "table"),
    [
        ("late deliveries by courier", "shipments"),
        ("refund rate by division", "refunds"),
        ("bKash payment success", "payments"),
        ("stockout at warehouse Gazipur-2", "inventory_snapshots"),
        ("seller tier history", "seller_tier_history"),
        ("promo code redemptions", "promotion_redemptions"),
        ("web session conversion", "web_sessions"),
        ("average product rating", "product_reviews"),
    ],
)
def test_retriever_finds_table(retr, q, table):
    assert table in [h.table for h in retr.search(q, 10)]


def test_role_filter():
    r = SchemaRetriever("analyst", use_embeddings=False)
    assert "support_messages" not in r.tables
    assert "email" not in r.tables["customers"]["columns"]
    s = SchemaRetriever("support", use_embeddings=False)
    assert "support_messages" in s.tables


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 2) == 0.5
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 3) == 1.0
    assert tokenize("Deliveries_by couriers") == ["delivery", "courier"]


def test_configs_load_and_judge_independence():
    for name in ("a_local", "b_mistral", "c_cascade", "d_gemini", "ci"):
        assert load_config(name).name == name
    with pytest.raises(ValueError, match="rule 14"):
        AgentConfig(
            name="bad",
            agent={"provider": "gemini", "model": "gemini-flash-latest"},
            judge={"provider": "gemini", "model": "gemini-flash-latest"},
        )
