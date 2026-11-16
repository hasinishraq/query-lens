import pytest
import yaml

from bazaar_world.mysqldb import admin_connect, fetchall
from bazaar_world.plant import FACTS_PATH, N_INJECTED_MESSAGES, N_INJECTED_REVIEWS, evaluate_facts
from bazaar_world.pools import INJECTIONS

FACT_IDS = [f"F0{i}" for i in range(1, 9)]


def _facts_yaml():
    with open(FACTS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def small_results(small_con):
    return evaluate_facts(small_con, "small")


@pytest.mark.parametrize("fid", FACT_IDS)
def test_fact_holds_small(small_results, fid):
    assert small_results[fid]["holds"], small_results[fid]


@pytest.mark.parametrize("fid", FACT_IDS)
def test_fact_matches_facts_yaml_small(small_results, fid):
    stored = {p["id"]: p for p in _facts_yaml()["planted"]}[fid]["expected"]["small"]
    assert stored["value"] == small_results[fid]["value"]


@pytest.mark.full
@pytest.mark.parametrize("fid", FACT_IDS)
def test_fact_holds_full(full_db, fid):
    con = admin_connect(full_db)
    try:
        res = evaluate_facts(con, "full")[fid]
    finally:
        con.close()
    assert res["holds"], res
    stored = {p["id"]: p for p in _facts_yaml()["planted"]}[fid]["expected"]["full"]
    assert stored["value"] == res["value"]


def test_injected_rows(small_con):
    inj = _facts_yaml()["traps"]["injected_rows"]["small"]
    assert len(inj["product_reviews"]) == N_INJECTED_REVIEWS
    assert len(inj["support_messages"]) == N_INJECTED_MESSAGES
    ids = ",".join(map(str, inj["product_reviews"]))
    bodies = [
        r[0] for r in fetchall(small_con, f"SELECT body FROM product_reviews WHERE review_id IN ({ids})")
    ]
    assert all(any(x in b for x in INJECTIONS) for b in bodies)
    ids = ",".join(map(str, inj["support_messages"]))
    msgs = [
        r[0] for r in fetchall(small_con, f"SELECT body FROM support_messages WHERE message_id IN ({ids})")
    ]
    assert all(m in INJECTIONS for m in msgs)
