from bazaar_world.dictionary_check import check, load_dictionary
from bazaar_world.schema import TABLES


def test_dictionary_matches_db(small_db):
    assert check(small_db, load_dictionary()) == []


def test_every_column_documented():
    d = load_dictionary()["tables"]
    for t in TABLES:
        for c, _ in t.columns:
            assert d[t.name]["columns"][c]["description"].strip(), (t.name, c)


def test_pii_and_untrusted_flags():
    d = load_dictionary()["tables"]
    assert d["customers"]["columns"]["email"]["pii"] is True
    assert d["support_messages"]["columns"]["body"]["untrusted"] is True
    assert "pii" not in d["orders"]["columns"]["order_id"]
