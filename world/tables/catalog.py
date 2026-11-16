"""Catalog: categories (fixed), brands, products, variants, price history (SCD2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import DAY, FAR_FUTURE, World
from ..pools import (
    ADJECTIVES,
    BRAND_COUNTRIES,
    BRAND_SYLLABLES,
    CATEGORY_PRICE,
    CATEGORY_TREE,
    COLOURS,
    SIZES,
)
from ..rng import allocate
from .sellers import PADMA

CLASSIC_PANJABI = "Classic Panjabi"


def build_categories(w: World) -> None:
    rows = []
    cid = 1
    l1_ids = {}
    for l1 in CATEGORY_TREE:
        rows.append((cid, None, l1, 1, l1))
        l1_ids[l1] = cid
        cid += 1
    l2_ids = {}
    for l1, subs in CATEGORY_TREE.items():
        for l2 in subs:
            rows.append((cid, l1_ids[l1], l2, 2, f"{l1} > {l2}"))
            l2_ids[(l1, l2)] = cid
            cid += 1
    leaf_meta = []
    for l1, subs in CATEGORY_TREE.items():
        for l2, leaves in subs.items():
            for leaf in leaves:
                rows.append((cid, l2_ids[(l1, l2)], leaf, 3, f"{l1} > {l2} > {leaf}"))
                leaf_meta.append((cid, l1, l2, leaf))
                cid += 1
    df = pd.DataFrame(rows, columns=["category_id", "parent_category_id", "name", "level", "path"])
    df["parent_category_id"] = df["parent_category_id"].astype("Int64")
    w.tables["categories"] = df
    w.meta["leaf_meta"] = leaf_meta


def build_brands(w: World) -> None:
    rng = w.rng("brands", "attrs")
    n = w.n("brands")
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < n:
        k = int(rng.integers(2, 4))
        nm = "".join(rng.choice(BRAND_SYLLABLES, k)).capitalize()
        if nm not in seen:
            seen.add(nm)
            names.append(nm)
    countries = list(BRAND_COUNTRIES)
    cp = np.array(list(BRAND_COUNTRIES.values()))
    country = rng.choice(countries, size=n, p=cp / cp.sum())
    w.tables["brands"] = pd.DataFrame(
        {
            "brand_id": np.arange(1, n + 1),
            "name": names,
            "country": country,
            "is_local": country == "Bangladesh",
        }
    )


def build_products(w: World) -> None:
    rng = w.rng("products", "attrs")
    rng_name = w.rng("products", "names")
    n = w.n("products")
    sellers = w.tables["sellers"]
    pop = w.meta["seller_pop"]
    main_cat = w.meta["seller_main_cat"]
    n_s = len(sellers)
    # each seller gets products in proportion to sqrt(popularity), at least 2
    per_seller = allocate(n - 2 * n_s, np.sqrt(pop)) + 2
    seller_idx = np.repeat(np.arange(n_s), per_seller)
    leaf_meta = w.meta["leaf_meta"]
    leaves_by_l1: dict[str, list[int]] = {}
    for i, (_, l1, _, _) in enumerate(leaf_meta):
        leaves_by_l1.setdefault(l1, []).append(i)
    leaf_idx = np.empty(n, dtype=int)
    for i, si in enumerate(seller_idx):
        if rng.random() < 0.8:
            opts = leaves_by_l1[main_cat[si]]
            leaf_idx[i] = opts[int(rng.integers(0, len(opts)))]
        else:
            leaf_idx[i] = int(rng.integers(0, len(leaf_meta)))
    cat_id = np.array([leaf_meta[j][0] for j in leaf_idx])
    l1s = [leaf_meta[j][1] for j in leaf_idx]
    leaf_names = [leaf_meta[j][3] for j in leaf_idx]
    n_b = w.n("brands")
    brand = rng.integers(1, n_b + 1, n)
    brand_names = w.tables["brands"]["name"].to_numpy()
    adj = rng_name.choice(ADJECTIVES, n)
    model = rng_name.integers(1, 999, n)
    names = np.array(
        [
            f"{brand_names[b - 1]} {a} {lf} {m}"
            for b, a, lf, m in zip(brand, adj, leaf_names, model, strict=True)
        ],
        dtype=object,
    )
    # duplicate names (1.5%): copy another product's name onto a product of a different seller
    n_dup = round(n * w.params["catalog"]["duplicate_name_share"])
    src = rng.choice(n, size=n_dup, replace=False)
    dst = rng.choice(n, size=n_dup, replace=False)
    for a, b in zip(src, dst, strict=True):
        if seller_idx[a] != seller_idx[b] and a != b:
            names[b] = names[a]
    # F03: two different products named exactly "Classic Panjabi", from different sellers
    panjabi_leaf = next(i for i, m in enumerate(leaf_meta) if m[3] == "Panjabi")
    fashion_sellers = [i for i in range(n_s) if main_cat[i] == "Fashion" and sellers["name"].iloc[i] != PADMA]
    chosen = fashion_sellers[:2] if len(fashion_sellers) >= 2 else [0, 2]
    planted_rows = []
    for si in chosen:
        row = int(np.flatnonzero(seller_idx == si)[0])
        names[row] = CLASSIC_PANJABI
        leaf_idx[row] = panjabi_leaf
        cat_id[row] = leaf_meta[panjabi_leaf][0]
        l1s[row] = "Fashion"
        leaf_names[row] = "Panjabi"
        planted_rows.append(row)
    # every other product with the planted name gets renamed so exactly two exist
    for i in range(n):
        if names[i] == CLASSIC_PANJABI and i not in planted_rows:
            names[i] = f"{CLASSIC_PANJABI} {int(model[i])}"
    lo = np.array([CATEGORY_PRICE[c][0] for c in l1s])
    hi = np.array([CATEGORY_PRICE[c][1] for c in l1s])
    # log-price skewed towards the cheap end of each department's range
    pos = rng.beta(1.2, 3.4, n)
    price = np.maximum(np.round(np.exp(np.log(lo) + pos * (np.log(hi) - np.log(lo))), -1), 30.0)
    joined = sellers["joined_at"].to_numpy().astype("datetime64[D]")[seller_idx]
    span = ((w.end - 60 * DAY) - joined) / DAY
    created_day = joined + (rng.beta(0.9, 2.2, n) * np.maximum(span, 1)).astype(int) * DAY
    created = created_day.astype("datetime64[s]") + (rng.integers(8 * 3600, 22 * 3600, n)).astype(
        "timedelta64[s]"
    )
    deleted = rng.random(n) < w.params["catalog"]["deleted_share"]
    deleted[planted_rows] = False
    status = np.where(deleted, "archived", np.where(rng.random(n) < 0.93, "active", "inactive"))
    df = pd.DataFrame(
        {
            "product_id": np.arange(1, n + 1),
            "seller_id": seller_idx + 1,
            "brand_id": brand,
            "category_id": cat_id,
            "name": names,
            "description": [
                f"{lf} by {brand_names[b - 1]}. {a} quality, ideal for everyday use."
                for lf, b, a in zip(leaf_names, brand, adj, strict=True)
            ],
            "list_price_bdt": price,
            "created_at": created,
            "status": status,
            "is_deleted": deleted,
            "weight_g": np.round(rng.lognormal(6.2, 1.0, n)).astype(int),
            "is_imported": rng.random(n) < w.params["catalog"]["imported_share"],
        }
    )
    w.tables["products"] = df
    w.meta["product_l1"] = np.array(l1s, dtype=object)
    # product popularity for orders: zipf-like within seller
    w.meta["product_pop"] = rng.pareto(1.3, n) + 0.2


def build_variants(w: World) -> None:
    rng = w.rng("product_variants", "attrs")
    prods = w.tables["products"]
    n_p = len(prods)
    n = w.n("product_variants")
    per = allocate(n - n_p, rng.gamma(1.0, 1.0, n_p) + 0.1) + 1
    pid = np.repeat(prods["product_id"].to_numpy(), per)
    l1 = np.repeat(w.meta["product_l1"], per)
    apparel = np.isin(l1, ["Fashion"])
    size = np.where(apparel, rng.choice(SIZES, n), None)
    colour = rng.choice(COLOURS, n)
    delta = np.where(
        rng.random(n) < 0.6,
        0.0,
        np.round(rng.normal(0, 0.06, n) * np.repeat(prods["list_price_bdt"].to_numpy(), per), -1),
    )
    w.tables["product_variants"] = pd.DataFrame(
        {
            "variant_id": np.arange(1, n + 1),
            "product_id": pid,
            "sku": [f"BZ-{p:05d}-{i:06d}" for p, i in zip(pid, np.arange(1, n + 1), strict=True)],
            "size": size,
            "colour": colour,
            "price_delta_bdt": delta,
            "barcode": [f"{x:013d}" for x in rng.integers(10**12, 10**13 - 1, n)],
            "is_active": rng.random(n) < 0.95,
        }
    )


def build_price_history(w: World) -> None:
    rng = w.rng("product_price_history", "walk")
    v = w.tables["product_variants"]
    prods = w.tables["products"].set_index("product_id")
    n_v = len(v)
    n = w.n("product_price_history")
    base = prods.loc[v["product_id"], "list_price_bdt"].to_numpy() + v["price_delta_bdt"].to_numpy()
    base = np.maximum(base, 20.0)
    created = prods.loc[v["product_id"], "created_at"].to_numpy().astype("datetime64[D]")
    first = np.maximum(created, w.start - 30 * DAY)
    span = ((w.end - first) / DAY).astype(int)
    cap = np.maximum(span // 40, 0)
    k = allocate(n - n_v, (rng.gamma(0.6, 1.0, n_v) + 0.02) * span)
    k = np.minimum(k, cap)
    deficit = (n - n_v) - int(k.sum())
    order = np.argsort(-(cap - k), kind="stable")
    j = 0
    while deficit > 0:
        idx = order[j % n_v]
        if k[idx] < cap[idx]:
            k[idx] += 1
            deficit -= 1
        j += 1
    reasons = ["promotion", "cost_increase", "competitor", "seasonal", "clearance"]
    vid, price, vf, vt, reason = [], [], [], [], []
    for i in range(n_v):
        ki = int(k[i])
        if ki:
            cuts = np.sort(rng.choice(np.arange(1, span[i]), size=ki, replace=False))
            dates = [created[i]] + [first[i] + int(c) * DAY for c in cuts]
        else:
            dates = [created[i]]
        p = base[i]
        for j2, d in enumerate(dates):
            if j2 > 0:
                p = max(20.0, round(p * float(np.clip(rng.normal(1.0, 0.07), 0.75, 1.3)), 0))
            vid.append(i + 1)
            price.append(round(float(p), 2))
            vf.append(d)
            vt.append(dates[j2 + 1] if j2 + 1 < len(dates) else FAR_FUTURE)
            reason.append("initial" if j2 == 0 else reasons[int(rng.integers(0, len(reasons)))])
    df = pd.DataFrame(
        {"variant_id": vid, "price_bdt": price, "valid_from": vf, "valid_to": vt, "change_reason": reason}
    )
    df.insert(0, "price_hist_id", np.arange(1, len(df) + 1))
    w.tables["product_price_history"] = df


def price_at(w: World, variant_id: np.ndarray, day: np.ndarray) -> np.ndarray:
    """SCD2 lookup: BDT price of each variant on each (Dhaka) day."""
    h = w.tables["product_price_history"]
    hv = h["variant_id"].to_numpy().astype(np.int64)
    hf = h["valid_from"].to_numpy().astype("datetime64[D]").astype(np.int64)
    key_h = hv * 1_000_000 + (hf + 200_000)
    order = np.argsort(key_h)
    key_h = key_h[order]
    prices = h["price_bdt"].to_numpy()[order]
    q = variant_id.astype(np.int64) * 1_000_000 + (day.astype("datetime64[D]").astype(np.int64) + 200_000)
    pos = np.searchsorted(key_h, q, side="right") - 1
    pos = np.clip(pos, 0, len(key_h) - 1)
    ok = (key_h[pos] // 1_000_000) == variant_id
    first_pos = np.searchsorted(key_h, variant_id.astype(np.int64) * 1_000_000, side="left")
    pos = np.where(ok, pos, first_pos)
    return prices[pos]


def build(w: World) -> None:
    build_brands(w)
    build_products(w)
    build_variants(w)
    build_price_history(w)
