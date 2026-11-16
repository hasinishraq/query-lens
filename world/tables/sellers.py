"""Sellers, seller tier history (SCD2) and seller payouts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import DAY, FAR_FUTURE, World
from ..pools import CATEGORY_TREE, RIVERS, SELLER_WORDS
from ..rng import allocate, pick_k
from .customers import bd_phone

PADMA = "Padma Gadgets"
MEGHNA = "Meghna Mart"
DUPLICATE_SELLER = "Jamuna Fashion"  # two sellers share this display name (duplicate-name trap)

WORD_CATEGORY = {
    "Gadgets": "Electronics",
    "Electronics": "Electronics",
    "Tech": "Electronics",
    "Fashion": "Fashion",
    "Boutique": "Fashion",
    "Collection": "Fashion",
    "Lifestyle": "Jewellery & Watches",
    "Kitchen": "Home & Living",
    "Home": "Home & Living",
    "Crafts": "Home & Living",
    "Organics": "Groceries",
    "Bazar": "Groceries",
    "Mart": "Groceries",
}


def build_sellers(w: World) -> None:
    rng = w.rng("sellers", "attrs")
    n = w.n("sellers")
    names: list[str] = [PADMA, MEGHNA, DUPLICATE_SELLER, DUPLICATE_SELLER]
    seen = set(names)
    while len(names) < n:
        nm = f"{rng.choice(RIVERS)} {rng.choice(SELLER_WORDS)}"
        if nm not in seen:
            seen.add(nm)
            names.append(nm)
    # shuffle everything except keep deterministic positions for planted sellers
    perm = rng.permutation(n)
    names_arr = np.array(names, dtype=object)[perm]
    l1 = list(CATEGORY_TREE)
    main_cat = []
    for nm in names_arr:
        word = str(nm).split(" ", 1)[1]
        main_cat.append(WORD_CATEGORY.get(word, l1[int(rng.integers(0, len(l1)))]))
    city = rng.choice(np.arange(1, 65), size=n, p=w.meta["city_weight"])
    # popularity: Pareto, with Padma Gadgets as one of the largest sellers (F01 needs volume)
    pop = rng.pareto(1.6, n) + 0.3
    is_padma = names_arr == PADMA
    pop[is_padma] = 0.0
    pop = pop / pop.sum() * (1 - 0.065)
    pop[is_padma] = 0.065
    is_meghna = names_arr == MEGHNA
    if pop[is_meghna][0] < 0.01:
        pop[is_meghna] = 0.012
    pop = pop / pop.sum()
    # joined_at chosen so that seller-months in the period equal the payout row target
    months_total = int(((w.end + DAY).astype("datetime64[M]") - w.start.astype("datetime64[M]")).astype(int))
    target = w.n("seller_payouts")
    months = np.full(n, months_total)
    reduce_by = months_total * n - target
    protected = is_padma | is_meghna | (names_arr == DUPLICATE_SELLER)
    candidates = rng.permutation(np.flatnonzero(~protected))
    for idx in candidates:
        if reduce_by <= 0:
            break
        r = int(min(reduce_by, rng.integers(1, months_total)))
        months[idx] -= r
        reduce_by -= r
    if reduce_by:
        raise RuntimeError("could not fit seller payout months")
    start_m = w.start.astype("datetime64[M]")
    join_month = start_m + (months_total - months)
    pre = months == months_total
    join_day = np.where(
        pre,
        np.datetime64("2021-01-01", "D") + rng.integers(0, 1090, n) * DAY,
        join_month.astype("datetime64[D]") + rng.integers(0, 27, n) * DAY,
    )
    joined = join_day.astype("datetime64[s]") + (rng.integers(9, 18, n) * 3600).astype("timedelta64[s]")
    slug = [str(s).lower().replace(" ", "") for s in names_arr]
    df = pd.DataFrame(
        {
            "seller_id": np.arange(1, n + 1),
            "name": names_arr,
            "legal_name": [f"{s} {'Ltd.' if rng.random() < 0.4 else 'Enterprise'}" for s in names_arr],
            "city_id": city,
            "joined_at": joined,
            "seller_type": np.where(
                rng.random(n) < 0.12, "brand_official", np.where(rng.random(n) < 0.5, "sme", "individual")
            ),
            "contact_email": [f"contact@{s}{i}.com.bd" for i, s in enumerate(slug, start=1)],
            "contact_phone": bd_phone(rng, n),
            "status": np.where(rng.random(n) < 0.94, "active", "suspended"),
            "rating_avg": np.round(np.clip(rng.normal(4.2, 0.35, n), 2.5, 5.0), 2),
        }
    )
    df.loc[is_padma | is_meghna, "status"] = "active"
    w.tables["sellers"] = df
    w.meta["seller_pop"] = pop
    w.meta["seller_main_cat"] = np.array(main_cat, dtype=object)


def build_tier_history(w: World) -> None:
    rng = w.rng("seller_tier_history", "walk")
    s = w.tables["sellers"]
    n_s = len(s)
    n = w.n("seller_tier_history")
    tiers = w.params["sellers"]["tiers"]
    names = s["name"].to_numpy()
    joined = s["joined_at"].to_numpy().astype("datetime64[D]")
    meghna = int(np.flatnonzero(names == MEGHNA)[0])
    n_changers = max(2, round(n_s * w.params["sellers"]["tier_change_share"]))
    early = joined < np.datetime64("2024-03-01", "D")
    others = np.array([i for i in range(n_s) if i != meghna and early[i]])
    changers = np.concatenate([[meghna], others[pick_k(rng, len(others), n_changers - 1)]])
    extra_rows = n - n_s
    alloc = np.zeros(n_s, dtype=int)
    alloc[meghna] = 1
    alloc[changers[1:]] = allocate(extra_rows - 1, np.ones(n_changers - 1))
    rows = []
    switch = np.datetime64("2025-07-01", "D")
    for i in range(n_s):
        if i == meghna:
            rows.append((i + 1, "silver", joined[i], switch))
            rows.append((i + 1, "gold", switch, FAR_FUTURE))
            continue
        start_from = max(joined[i], w.start)
        k = int(alloc[i])
        months = np.arange(start_from.astype("datetime64[M]") + 1, (w.end + DAY).astype("datetime64[M]"))
        k = min(k, len(months))
        alloc[i] = k
        cut = np.sort(rng.choice(months, size=k, replace=False)).astype("datetime64[D]") if k else []
        tier = tiers[int(rng.choice(4, p=[0.45, 0.32, 0.17, 0.06]))]
        dates = [joined[i], *cut]
        for j, d in enumerate(dates):
            if j > 0:
                ti = tiers.index(tier)
                step = -1 if ti == 3 else (1 if ti == 0 else int(rng.choice([-1, 1])))
                tier = tiers[ti + step]
            rows.append((i + 1, tier, d, dates[j + 1] if j + 1 < len(dates) else FAR_FUTURE))
    if len(rows) != n:
        raise RuntimeError(f"tier history has {len(rows)} rows, expected {n}")
    df = pd.DataFrame(rows, columns=["seller_id", "tier", "valid_from", "valid_to"])
    df.insert(0, "tier_hist_id", np.arange(1, len(df) + 1))
    w.tables["seller_tier_history"] = df


def build_payouts(w: World) -> None:
    """Monthly payout per seller from first active month; gross = delivered item value in that month."""
    s = w.tables["sellers"]
    items = w.tables["order_items"]
    orders = w.tables["orders"]
    tiers = w.tables["seller_tier_history"]
    comm = w.params["sellers"]["commission"]
    od = orders.set_index("order_id")
    ok = od["status"].isin(["delivered", "returned"]).to_numpy() & ~od["is_deleted"].to_numpy()
    good = od.index.to_numpy()[ok]
    it = items[items["order_id"].isin(good)]
    month = od.loc[it["order_id"], "order_at"].to_numpy().astype("datetime64[M]")
    agg = pd.DataFrame(
        {"seller_id": it["seller_id"].to_numpy(), "m": month, "v": it["line_total_bdt"].to_numpy()}
    )
    gross = agg.groupby(["seller_id", "m"])["v"].sum().to_dict()
    rows = []
    start_m = w.start.astype("datetime64[M]")
    end_m = w.end.astype("datetime64[M]")
    tier_by_seller: dict[int, list[tuple[np.datetime64, np.datetime64, str]]] = {}
    for r in tiers.itertuples(index=False):
        tier_by_seller.setdefault(int(r.seller_id), []).append(
            (np.datetime64(r.valid_from, "D"), np.datetime64(r.valid_to, "D"), str(r.tier))
        )
    for r in s.itertuples(index=False):
        jm = max(np.datetime64(r.joined_at, "M"), start_m)
        for m in np.arange(jm, end_m + 1):
            ps = m.astype("datetime64[D]")
            pe = (m + 1).astype("datetime64[D]") - DAY
            g = round(float(gross.get((int(r.seller_id), m), 0.0)), 2)
            tier = next((t for a, b, t in tier_by_seller[int(r.seller_id)] if a <= pe < b), "bronze")
            paid = (pe + 7 * DAY).astype("datetime64[s]") + np.timedelta64(11, "h")
            rows.append(
                (
                    int(r.seller_id),
                    ps,
                    pe,
                    g,
                    round(g * comm[tier], 2),
                    paid if pe + 7 * DAY <= w.end else np.datetime64("NaT", "s"),
                )
            )
    df = pd.DataFrame(
        rows, columns=["seller_id", "period_start", "period_end", "gross_bdt", "commission_bdt", "paid_at"]
    )
    df = df.sort_values(["period_start", "seller_id"], kind="stable").reset_index(drop=True)
    df.insert(0, "payout_id", np.arange(1, len(df) + 1))
    w.tables["seller_payouts"] = df
