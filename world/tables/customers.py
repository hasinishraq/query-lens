"""Customers, addresses and segment history (loyalty is built after orders)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import DAY, FAR_FUTURE, SECOND, World, probs, random_time_of_day
from ..pools import AREAS, EMAIL_DOMAINS, FIRST_NAMES_F, FIRST_NAMES_M, LAST_NAMES, street_pool
from ..rng import allocate, choice

BOISHAKH25 = "Boishakh25"
BOISHAKH25_COHORT = {"small": 12, "full": 70}

REGION_POSTCODE_BASE = {1: 1000, 2: 4000, 3: 6000, 4: 9000, 5: 8200, 6: 3100, 7: 5400, 8: 2200}


def bd_phone(rng: np.random.Generator, n: int) -> np.ndarray:
    op = rng.choice(["13", "14", "15", "16", "17", "18", "19"], size=n)
    num = rng.integers(0, 100_000_000, size=n)
    intl = rng.random(n) < 0.6
    return np.array(
        [("+880" if i else "0") + f"{o}{x:08d}" for o, x, i in zip(op, num, intl, strict=True)], dtype=object
    )


def build_customers(w: World) -> None:
    p = w.params["customers"]
    n = w.n("customers")
    rng_n = w.rng("customers", "names")
    rng_a = w.rng("customers", "attrs")
    rng_s = w.rng("customers", "signup")

    gkeys, gp = probs(p["gender"])
    gender = choice(rng_a, gkeys, gp, n)
    first = np.where(gender == "female", rng_n.choice(FIRST_NAMES_F, n), rng_n.choice(FIRST_NAMES_M, n))
    last = rng_n.choice(LAST_NAMES, n)
    full = np.array([f"{a} {b}" for a, b in zip(first, last, strict=True)], dtype=object)
    dom = rng_n.choice(EMAIL_DOMAINS, n, p=[0.62, 0.14, 0.1, 0.08, 0.06])
    num = rng_n.integers(1, 9999, n)
    email = np.array(
        [f"{a.lower()}.{b.lower()}{k}@{d}" for a, b, k, d in zip(first, last, num, dom, strict=True)],
        dtype=object,
    )

    city = rng_a.choice(np.arange(1, 65), size=n, p=w.meta["city_weight"])

    # signups: a share before the period, the rest following order-volume growth
    pre = rng_s.random(n) < p["signup_before_period_share"]
    earliest = np.datetime64(str(p["signup_earliest"]), "D")
    pre_days = int((w.start - earliest) / DAY)
    dw = w.meta["day_weight"] / w.meta["day_weight"].sum()
    in_day = rng_s.choice(len(w.days), size=n, p=dw)
    sday = np.where(pre, earliest + rng_s.integers(0, pre_days, n) * DAY, w.days[in_day])
    # F06: a fixed cohort signs up during the Boishakh25 campaign and is acquired by it
    camp_all = w.tables["marketing_campaigns"]
    b25 = camp_all.loc[camp_all["name"] == BOISHAKH25].iloc[0]
    b_start = np.datetime64(b25["start_date"], "D")
    b_days = int((np.datetime64(b25["end_date"], "D") - b_start) / DAY) + 1
    n_b25 = BOISHAKH25_COHORT[w.size]
    rng_b = w.rng("customers", "boishakh25")
    cohort = rng_b.choice(np.flatnonzero(~pre), size=n_b25, replace=False)
    sday[cohort] = b_start + rng_b.integers(0, b_days, n_b25) * DAY
    forced = np.zeros(n, dtype=bool)
    forced[cohort] = True
    signup = sday.astype("datetime64[s]") + random_time_of_day(rng_s, n, w.params["orders"]["hour_weights"])
    # sort customers by signup so ids grow with time
    order = np.argsort(signup, kind="stable")
    signup = signup[order]
    forced = forced[order]

    ckeys, cp = probs(p["signup_channel"])
    akeys, ap = probs(p["acquisition_source"])
    acq = choice(rng_a, akeys, ap, n).astype(object)
    camp = w.tables["marketing_campaigns"]
    cs = camp["start_date"].to_numpy().astype("datetime64[D]")
    ce = camp["end_date"].to_numpy().astype("datetime64[D]")
    cnames = camp["name"].to_numpy()
    is_acq = camp["objective"].to_numpy() == "acquisition"
    sd = signup.astype("datetime64[D]")
    for i in np.flatnonzero(acq == "campaign"):
        active = np.flatnonzero((cs <= sd[i]) & (ce >= sd[i]) & is_acq)
        if len(active) == 0:
            active = np.flatnonzero((cs <= sd[i]) & (ce >= sd[i]))
        acq[i] = cnames[rng_a.choice(active)] if len(active) else "organic"
    acq[acq == BOISHAKH25] = "organic"  # only the planted cohort carries this campaign
    acq[forced] = BOISHAKH25

    lkeys, lp = probs(p["preferred_language"])
    deleted = rng_a.random(n) < p["deleted_share"]
    upd = signup + (rng_a.integers(0, 400, n) * 86_400).astype("timedelta64[s]")
    upd = np.minimum(upd, (w.end + DAY).astype("datetime64[s]") - SECOND)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "full_name": full,
            "email": email,
            "phone": bd_phone(rng_n, n),
            "gender": gender,
            "birth_year": rng_a.integers(1960, 2007, n),
            "city_id": city,
            "signup_at": signup,
            "signup_channel": choice(rng_a, ckeys, cp, n),
            "acquisition_source": acq,
            "is_verified": rng_a.random(n) < p["verified_share"],
            "preferred_language": choice(rng_a, lkeys, lp, n),
            "is_deleted": deleted,
            "updated_at": upd,
        }
    )
    w.tables["customers"] = df


def build_addresses(w: World) -> None:
    rng = w.rng("customer_addresses", "attrs")
    cust = w.tables["customers"]
    n_c = len(cust)
    n = w.n("customer_addresses")
    extra = n - n_c
    owners_extra = np.sort(rng.choice(n_c, size=extra, replace=True))
    owner = np.concatenate([np.arange(n_c), owners_extra])
    is_default = np.concatenate([np.ones(n_c, bool), np.zeros(extra, bool)])
    order = np.lexsort((~is_default, owner))
    owner, is_default = owner[order], is_default[order]
    home_city = cust["city_id"].to_numpy()[owner]
    other_city = rng.choice(np.arange(1, 65), size=n, p=w.meta["city_weight"])
    move = (~is_default) & (rng.random(n) < 0.3)
    city = np.where(move, other_city, home_city)
    regions = w.tables["cities"].set_index("city_id")["region_id"].to_dict()
    streets = street_pool(w.seed)
    st = rng.choice(streets, n)
    road = rng.integers(1, 30, n)
    area = rng.choice(AREAS, n)
    street = np.array(
        [f"House {s}, Road {r}, {a}" for s, r, a in zip(st, road, area, strict=True)], dtype=object
    )
    post = np.array(
        [
            str(REGION_POSTCODE_BASE[regions[c]] + int(x))
            for c, x in zip(city, rng.integers(0, 300, n), strict=True)
        ],
        dtype=object,
    )
    signup = cust["signup_at"].to_numpy()[owner]
    created = signup + np.where(is_default, 0, rng.integers(1, 500, n) * 86_400).astype("timedelta64[s]")
    created = np.minimum(created, (w.end + DAY).astype("datetime64[s]") - SECOND)
    df = pd.DataFrame(
        {
            "address_id": np.arange(1, n + 1),
            "customer_id": owner + 1,
            "city_id": city,
            "street_address": street,
            "postcode": post,
            "address_type": np.where(
                rng.random(n) < 0.8, "home", np.where(rng.random(n) < 0.7, "office", "other")
            ),
            "is_default": is_default,
            "created_at": created,
            "is_deleted": (~is_default) & (rng.random(n) < 0.06),
        }
    )
    w.tables["customer_addresses"] = df


def build_segments(w: World) -> None:
    rng = w.rng("customer_segment_history", "walk")
    cust = w.tables["customers"]
    n_c = len(cust)
    n = w.n("customer_segment_history")
    segs = w.params["customers"]["segments"]
    signup = cust["signup_at"].to_numpy().astype("datetime64[D]")
    spans = ((w.end - signup) / DAY).astype(int)
    changes = allocate(n - n_c, (rng.gamma(1.2, 1.0, n_c) + 0.05) * (spans + 1))
    cap = np.maximum(spans // 30, 0)  # at most one change per ~month
    over = np.maximum(changes - cap, 0)
    changes = np.minimum(changes, cap)
    deficit = int(over.sum())
    room_order = np.argsort(-(cap - changes), kind="stable")
    j = 0
    while deficit > 0:
        idx = room_order[j % n_c]
        if changes[idx] < cap[idx]:
            changes[idx] += 1
            deficit -= 1
        j += 1
    rows_c, rows_s, rows_f, rows_t = [], [], [], []
    trans = {
        "new": ["regular", "at_risk"],
        "regular": ["loyal", "at_risk", "vip"],
        "loyal": ["vip", "regular", "at_risk"],
        "vip": ["loyal", "at_risk"],
        "at_risk": ["regular", "churned"],
        "churned": ["regular", "at_risk"],
    }
    end = w.end
    for i in range(n_c):
        k = int(changes[i])
        start = signup[i]
        span = int((end - start) / DAY)
        if k > 0 and span > k:
            cut = np.sort(rng.choice(np.arange(1, span), size=k, replace=False))
            dates = [start] + [start + int(c) * DAY for c in cut]
        else:
            dates = [start]
            k = 0
        seg = "new"
        for j, d in enumerate(dates):
            if j > 0:
                opts = trans[seg]
                seg = opts[int(rng.integers(0, len(opts)))]
            rows_c.append(i + 1)
            rows_s.append(seg)
            rows_f.append(d)
            rows_t.append(dates[j + 1] if j + 1 < len(dates) else FAR_FUTURE)
    # top up if any customer could not take its share of changes
    df = pd.DataFrame({"customer_id": rows_c, "segment": rows_s, "valid_from": rows_f, "valid_to": rows_t})
    short = n - len(df)
    if short:
        raise RuntimeError(f"segment history short by {short} rows")
    df.insert(0, "segment_hist_id", np.arange(1, len(df) + 1))
    df["is_current"] = df["valid_to"].to_numpy().astype("datetime64[D]") == FAR_FUTURE
    w.tables["customer_segment_history"] = df
    _ = segs


def build(w: World) -> None:
    build_customers(w)
    build_addresses(w)
    build_segments(w)
