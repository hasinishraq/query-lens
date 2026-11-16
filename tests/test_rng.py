import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from bazaar_world.rng import allocate, exact_choice, pick_k, rng_for, stable_hash


def test_same_inputs_same_stream():
    a = rng_for(42, "orders", "dates").random(5)
    b = rng_for(42, "orders", "dates").random(5)
    assert np.array_equal(a, b)


def test_purposes_are_independent():
    a = rng_for(42, "orders", "dates").random(5)
    b = rng_for(42, "orders", "status").random(5)
    c = rng_for(43, "orders", "dates").random(5)
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stable_hash_is_stable():
    assert stable_hash(42, "orders", "dates") == stable_hash(42, "orders", "dates")
    assert stable_hash(42, "a") != stable_hash(42, "b")


@given(st.integers(0, 10_000), st.lists(st.floats(0.01, 100), min_size=1, max_size=30))
def test_allocate_sums_to_total(total, weights):
    out = allocate(total, weights)
    assert out.sum() == total
    assert (out >= 0).all()


def test_allocate_is_proportional():
    assert allocate(1000, [1, 3]).tolist() == [250, 750]


def test_exact_choice_counts():
    vals = exact_choice(rng_for(1, "t", "p"), ["a", "b", "c"], [0.5, 0.3, 0.2], 1000)
    assert (vals == "a").sum() == 500
    assert (vals == "b").sum() == 300
    assert (vals == "c").sum() == 200


def test_rng_distribution_sanity():
    x = rng_for(7, "t", "u").random(20_000)
    assert abs(x.mean() - 0.5) < 0.01
    assert abs(x.std() - (1 / 12) ** 0.5) < 0.01


def test_pick_k_distinct():
    idx = pick_k(rng_for(1, "t", "k"), 100, 10)
    assert len(set(idx.tolist())) == 10
