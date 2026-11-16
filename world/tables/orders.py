"""Orders, order items (with promotion redemptions) and order status events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import DAY, SECOND, World, probs, random_time_of_day
from ..rng import allocate, exact_choice
from .catalog import price_at

VALID_EXCLUDED = ("cancelled", "canceled", "payment_failed")
SHIPPED_STATUSES = ("delivered", "in_transit", "returned")
BOISHAKH = "Boishakh25"


def _order_counts(w: World) -> np.ndarray:
    p = w.params["customers"]
    rng = w.rng("orders", "counts")
    cust = w.tables["customers"]
    n_c = len(cust)
    n_o = w.n("orders")
    signup_day = cust["signup_at"].to_numpy().astype("datetime64[D]")
    dw = w.meta["day_weight"]
    cw = np.concatenate([[0.0], np.cumsum(dw)])
    s_idx = np.clip(((signup_day - w.start) / DAY).astype(int), 0, len(dw) - 1)
    exposure = cw[-1] - cw[s_idx]
    activity = rng.lognormal(0.0, p["activity_sigma"], n_c)
    single = np.zeros(n_c, dtype=bool)
    n_single = round(n_c * p["one_order_share"])
    single[rng.choice(n_c, size=n_single, replace=False)] = True
    acq = cust["acquisition_source"].to_numpy()
    boishakh = acq == BOISHAKH
    single[boishakh] = False  # planted F06 cohort repeats
    multi = ~single
    counts = np.ones(n_c, dtype=np.int64)
    counts[multi] = 2
    remaining = n_o - int(counts.sum())
    counts[multi] += allocate(remaining, (activity * exposure)[multi])
    # F06 cohort: at least 3 orders each (taken from the heaviest customers to keep the total)
    lack = np.clip(3 - counts[boishakh], 0, None)
    counts[boishakh] += lack
    donors = np.argsort(-counts, kind="stable")
    k = int(lack.sum())
    j = 0
    while k > 0:
        if not boishakh[donors[j]]:
            counts[donors[j]] -= 1
            k -= 1
        j += 1
    w.meta["customer_exposure"] = exposure
    return counts


def _sample_days(w: World, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order day (index into w.days) for each order, and its customer index."""
    rng = w.rng("orders", "dates")
    cust = w.tables["customers"]
    signup_day = cust["signup_at"].to_numpy().astype("datetime64[D]")
    dw = w.meta["day_weight"]
    s_idx = np.clip(((signup_day - w.start) / DAY).astype(int), 0, len(dw) - 1)
    owner = np.repeat(np.arange(len(counts)), counts)
    # Sampling after each customer's signup over-weights late days (signups also grow), so the sampling
    # weights are corrected per month until monthly volume follows the target growth curve.
    month = w.meta["month_of_day"]
    n_months = int(month.max()) + 1
    target_m = np.bincount(month, weights=dw, minlength=n_months)
    target_m = target_m / target_m.sum()
    q = dw.copy()
    u0 = rng.random(len(owner))
    day = np.zeros(len(owner), dtype=np.int64)
    for _ in range(6):
        cq = np.concatenate([[0.0], np.cumsum(q)])
        lo = cq[s_idx[owner]]
        u = lo + u0 * (cq[-1] - lo)
        day = np.clip(np.searchsorted(cq, u, side="right") - 1, 0, len(dw) - 1)
        got = np.bincount(month[day], minlength=n_months) / len(day)
        q = q * (target_m / np.maximum(got, 1e-9))[month]
    # F06: most Boishakh25-acquired customers place a second order within 90 days of the first
    boishakh = np.flatnonzero(cust["acquisition_source"].to_numpy() == BOISHAKH)
    rng_b = w.rng("orders", "boishakh25")
    starts = np.concatenate([[0], np.cumsum(counts)])
    for ci in boishakh:
        a, b = starts[ci], starts[ci + 1]
        seg = np.sort(day[a:b])
        if len(seg) >= 3:
            seg[1] = min(seg[0] + int(rng_b.integers(5, 45)), len(dw) - 1)
            seg[2] = min(seg[1] + int(rng_b.integers(3, 40)), len(dw) - 1)
            day[a:b] = np.sort(seg)
    return day, owner


def _plant_1111(w: World, day: np.ndarray, owner: np.ndarray, margin: float) -> np.ndarray:
    """F05: make 11.11 2025 the busiest day by a clear margin (moves a few orders onto that day)."""
    rng = w.rng("orders", "plant_1111")
    target_day = int((np.datetime64("2025-11-11", "D") - w.start) / DAY)
    counts = np.bincount(day, minlength=len(w.days))
    other_max = int(np.max(np.delete(counts, target_day)))
    need = int(np.ceil(other_max * margin)) - int(counts[target_day])
    if need <= 0:
        return day
    signup_day = w.tables["customers"]["signup_at"].to_numpy().astype("datetime64[D]")
    ok_owner = signup_day[owner] < np.datetime64("2025-11-11", "D")
    hol = {int((d - w.start) / DAY) for d in w.meta["holidays"]}
    movable = np.flatnonzero(ok_owner & (day != target_day) & ~np.isin(day, list(hol)))
    pick = rng.choice(movable, size=need, replace=False)
    day = day.copy()
    day[pick] = target_day
    return day


def build_orders(w: World, margin: float | None = None) -> None:
    p = w.params["orders"]
    margin = float(p["sale_1111_2025_margin"]) if margin is None else margin
    rng = w.rng("orders", "attrs")
    counts = _order_counts(w)
    day, owner = _sample_days(w, counts)
    day = _plant_1111(w, day, owner, margin)
    n = len(day)
    ts = w.days[day].astype("datetime64[s]") + random_time_of_day(
        w.rng("orders", "time"), n, p["hour_weights"]
    )
    cust = w.tables["customers"]
    signup = cust["signup_at"].to_numpy()[owner]
    ts = np.maximum(ts, signup + np.timedelta64(600, "s"))
    ts = np.minimum(ts, (w.end + DAY).astype("datetime64[s]") - SECOND)
    order = np.lexsort((owner, ts))
    ts, owner = ts[order], owner[order]
    oday = ts.astype("datetime64[D]")

    # status: in_transit only for recent orders
    recent = oday > w.end - p["in_transit_days"] * DAY
    st = np.empty(n, dtype=object)
    sp = p["status"]
    old_keys = ["delivered", "cancelled", "returned", "payment_failed"]
    old_p = [sp["delivered"] + sp["in_transit"], sp["cancelled"], sp["returned"], sp["payment_failed"]]
    st[~recent] = exact_choice(w.rng("orders", "status_old"), old_keys, old_p, int((~recent).sum()))
    st[recent] = exact_choice(
        w.rng("orders", "status_recent"),
        ["in_transit", "cancelled", "payment_failed"],
        [sp["in_transit"] + sp["delivered"] + sp["returned"], sp["cancelled"], sp["payment_failed"]],
        int(recent.sum()),
    )
    canc = np.flatnonzero(st == "cancelled")
    ck, cp = probs(p["cancel_spelling"])
    st[canc] = exact_choice(w.rng("orders", "cancel_spelling"), ck, cp, len(canc))

    chk, chp = probs(p["channel"])
    channel = exact_choice(w.rng("orders", "channel"), chk, chp, n)
    device = np.empty(n, dtype=object)
    rng_d = w.rng("orders", "device")
    for ch, dist in p["device_by_channel"].items():
        m = channel == ch
        dk, dp = probs(dist)
        device[m] = rng_d.choice(np.array(dk, dtype=object), size=int(m.sum()), p=np.array(dp) / sum(dp))
    usd = w.rng("orders", "currency").random(n) < p["usd_share"]
    usd &= channel != "social_commerce"

    # address: default address mostly; another live address of the customer otherwise
    addr = w.tables["customer_addresses"]
    a_c = addr["customer_id"].to_numpy() - 1
    a_id = addr["address_id"].to_numpy()
    a_def = addr["is_default"].to_numpy()
    a_del = addr["is_deleted"].to_numpy()
    a_created = addr["created_at"].to_numpy()
    default_of = np.zeros(len(cust), dtype=np.int64)
    default_of[a_c[a_def]] = a_id[a_def]
    others: dict[int, list[int]] = {}
    for i in np.flatnonzero(~a_def & ~a_del):
        others.setdefault(int(a_c[i]), []).append(i)
    address = default_of[owner].copy()
    rng_a = w.rng("orders", "address")
    alt = rng_a.random(n) < 0.15
    for i in np.flatnonzero(alt):
        opts = [j for j in others.get(int(owner[i]), []) if a_created[j] <= ts[i]]
        if opts:
            address[i] = a_id[opts[int(rng_a.integers(0, len(opts)))]]

    deleted = rng.random(n) < p["deleted_share"]
    upd = ts + (rng.integers(3600, 20 * 86400, n)).astype("timedelta64[s]")
    upd = np.minimum(upd, (w.end + DAY).astype("datetime64[s]") - SECOND)
    df = pd.DataFrame(
        {
            "order_id": np.arange(1, n + 1),
            "customer_id": owner + 1,
            "address_id": address,
            "order_at": ts,
            "status": st,
            "channel": channel,
            "currency": np.where(usd, "USD", "BDT"),
            "total_amount": 0.0,
            "total_bdt": 0.0,
            "promo_id": pd.array([pd.NA] * n, dtype="Int64"),
            "is_deleted": deleted,
            "updated_at": upd,
            "device_type": device,
        }
    )
    w.tables["orders"] = df


def _seller_tables(w: World) -> dict[str, object]:
    prods = w.tables["products"]
    var = w.tables["product_variants"]
    sellers = w.tables["sellers"]
    n_s = len(sellers)
    created = prods["created_at"].to_numpy().astype("datetime64[D]")
    pop = w.meta["product_pop"]
    var_by_prod: dict[int, np.ndarray] = {}
    vp = var["product_id"].to_numpy()
    vid = var["variant_id"].to_numpy()
    starts = np.searchsorted(vp, np.arange(1, len(prods) + 2))
    for pid in range(1, len(prods) + 1):
        var_by_prod[pid] = vid[starts[pid - 1] : starts[pid]]
    sp = prods["seller_id"].to_numpy() - 1
    per_seller = []
    first_prod = np.full(n_s, np.datetime64("9999-12-31", "D"))
    for s in range(n_s):
        idx = np.flatnonzero(sp == s)
        idx = idx[np.argsort(created[idx], kind="stable")]
        per_seller.append((idx, created[idx], pop[idx]))
        if len(idx):
            first_prod[s] = created[idx[0]]
    return {"per_seller": per_seller, "first_prod": first_prod, "var_by_prod": var_by_prod}


def build_items(w: World) -> None:
    p = w.params["orders"]
    o = w.tables["orders"]
    n_o = len(o)
    rng = w.rng("order_items", "pick")
    n_items = w.n("order_items")
    ik = [int(k) for k in p["items_per_order"]]
    ip = list(p["items_per_order"].values())
    k_per = exact_choice(w.rng("order_items", "count"), ik, ip, n_o).astype(np.int64)
    # adjust to the exact item total, keeping each order within 1..6 items
    diff = n_items - int(k_per.sum())
    rng_k = w.rng("order_items", "count_fix")
    while diff != 0:
        i = int(rng_k.integers(0, n_o))
        if diff > 0 and k_per[i] < 6:
            k_per[i] += 1
            diff -= 1
        elif diff < 0 and k_per[i] > 1:
            k_per[i] -= 1
            diff += 1
    status = o["status"].to_numpy()
    shipped = np.isin(status, SHIPPED_STATUSES)
    extra_ship = w.n("shipments") - int(shipped.sum())
    cand = np.flatnonzero(shipped & (k_per >= 2))
    two_seller = np.zeros(n_o, dtype=bool)
    if extra_ship > len(cand):
        raise RuntimeError("not enough multi-item orders for split shipments")
    if extra_ship > 0:
        two_seller[w.rng("order_items", "split").choice(cand, size=extra_ship, replace=False)] = True
    # a few unshipped orders also have two sellers (realism; they never ship)
    uns = np.flatnonzero(~shipped & (k_per >= 2))
    two_seller[uns[w.rng("order_items", "split_u").random(len(uns)) < 0.12]] = True

    st = _seller_tables(w)
    per_seller = st["per_seller"]
    first_prod = st["first_prod"]
    var_by_prod = st["var_by_prod"]
    seller_pop = w.meta["seller_pop"]
    sellers = w.tables["sellers"]
    joined = sellers["joined_at"].to_numpy().astype("datetime64[D]")
    oday = o["order_at"].to_numpy().astype("datetime64[D]")

    def pick_seller(d: np.datetime64, exclude: int = -1) -> int:
        avail = (first_prod <= d) & (joined <= d)
        if exclude >= 0:
            avail[exclude] = False
        pw = seller_pop * avail
        return int(rng.choice(len(pw), p=pw / pw.sum()))

    def pick_variants(s: int, d: np.datetime64, k: int) -> list[int]:
        idx, cr, pp = per_seller[s]
        m = int(np.searchsorted(cr, d, side="right"))
        pw = pp[:m]
        prods = idx[rng.choice(m, size=k, p=pw / pw.sum())] + 1
        out = []
        for pid in prods:
            vs = var_by_prod[int(pid)]
            out.append(int(vs[int(rng.integers(0, len(vs)))]))
        return out

    item_order = np.repeat(np.arange(1, n_o + 1), k_per)
    item_var = np.empty(n_items, dtype=np.int64)
    item_seller = np.empty(n_items, dtype=np.int64)
    pos = 0
    for i in range(n_o):
        k = int(k_per[i])
        d = oday[i]
        s1 = pick_seller(d)
        if two_seller[i]:
            k1 = (k + 1) // 2
            s2 = pick_seller(d, exclude=s1)
            vs = pick_variants(s1, d, k1) + pick_variants(s2, d, k - k1)
            ss = [s1] * k1 + [s2] * (k - k1)
        else:
            vs = pick_variants(s1, d, k)
            ss = [s1] * k
        item_var[pos : pos + k] = vs
        item_seller[pos : pos + k] = ss
        pos += k
    qk = [int(x) for x in p["quantity"]]
    qty = exact_choice(w.rng("order_items", "qty"), qk, list(p["quantity"].values()), n_items).astype(
        np.int64
    )
    item_day = np.repeat(oday, k_per)
    price_bdt = price_at(w, item_var, item_day)
    rates = w.tables["currency_rates"].set_index("rate_date")["bdt_per_unit"]
    rate_by_day = rates.reindex(pd.to_datetime(w.days)).to_numpy()
    item_dayidx = ((item_day - w.start) / DAY).astype(int)
    usd_order = o["currency"].to_numpy() == "USD"
    item_usd = np.repeat(usd_order, k_per)
    rate = rate_by_day[item_dayidx]
    unit = np.where(item_usd, np.round(price_bdt / rate, 2), price_bdt)
    gross_bdt = np.where(item_usd, np.round(unit * qty * rate, 2), np.round(unit * qty, 2))

    # promotions: exactly n redemptions on BDT orders with an eligible active promo
    discount = np.zeros(n_items)
    promo_of = np.full(n_o, -1, dtype=np.int64)
    promos = w.tables["promotions"]
    ps = promos["start_at"].to_numpy()
    pe = promos["end_at"].to_numpy()
    pmin = promos["min_order_bdt"].to_numpy()
    ptype = promos["promo_type"].to_numpy()
    pval = promos["discount_value"].to_numpy()
    pmax = promos["max_redemptions"].to_numpy()
    used = np.zeros(len(promos), dtype=np.int64)
    item_starts = np.concatenate([[0], np.cumsum(k_per)])
    subtotal = np.add.reduceat(gross_bdt, item_starts[:-1])
    ots = o["order_at"].to_numpy()
    n_red = w.n("promotion_redemptions")
    rng_p = w.rng("promotion_redemptions", "assign")
    order_perm = rng_p.permutation(n_o)
    got = 0
    for i in order_perm:
        if got >= n_red:
            break
        if usd_order[i]:
            continue
        act = np.flatnonzero((ps <= ots[i]) & (pe >= ots[i]) & (pmin <= subtotal[i]) & (used < pmax))
        if len(act) == 0:
            continue
        j = int(act[int(rng_p.integers(0, len(act)))])
        used[j] += 1
        promo_of[i] = j
        got += 1
    if got < n_red:
        raise RuntimeError(f"only {got} promo redemptions possible, need {n_red}")
    for i in np.flatnonzero(promo_of >= 0):
        j = promo_of[i]
        a, b = item_starts[i], item_starts[i + 1]
        sub = subtotal[i]
        disc = sub * pval[j] / 100.0 if ptype[j] == "percent" else float(pval[j])
        disc = round(min(disc, sub * 0.5, 1500.0), 2)
        share = gross_bdt[a:b] / sub
        d_items = np.round(disc * share, 2)
        d_items[-1] = round(disc - d_items[:-1].sum(), 2)
        discount[a:b] = d_items
    line_total = np.round(gross_bdt - discount, 2)

    rng_g = w.rng("order_items", "gift")
    items = pd.DataFrame(
        {
            "order_item_id": np.arange(1, n_items + 1),
            "order_id": item_order,
            "variant_id": item_var,
            "seller_id": item_seller + 1,
            "quantity": qty,
            "unit_price": unit,
            "discount_bdt": discount,
            "line_total_bdt": line_total,
            "is_gift": rng_g.random(n_items) < p["gift_share"],
        }
    )
    w.tables["order_items"] = items
    total_bdt = np.round(np.add.reduceat(line_total, item_starts[:-1]), 2)
    total_usd = np.round(np.add.reduceat(np.round(unit * qty, 2), item_starts[:-1]), 2)
    o["total_amount"] = np.where(usd_order, total_usd, total_bdt)
    o["total_bdt"] = pd.array(np.where(usd_order, np.nan, total_bdt), dtype="Float64")
    o.loc[usd_order, "total_bdt"] = pd.NA
    pid = np.where(promo_of >= 0, promos["promo_id"].to_numpy()[np.maximum(promo_of, 0)], 0)
    o["promo_id"] = pd.array(np.where(promo_of >= 0, pid, 0), dtype="Int64")
    o.loc[promo_of < 0, "promo_id"] = pd.NA
    # BDT value of every order (used by payments): converted for USD orders
    w.meta["order_value_bdt"] = np.round(np.add.reduceat(line_total, item_starts[:-1]), 2)
    w.meta["order_rate"] = rate_by_day[((oday - w.start) / DAY).astype(int)]
    w.meta["item_starts"] = item_starts

    red_idx = np.flatnonzero(promo_of >= 0)
    disc_by_order = np.add.reduceat(discount, item_starts[:-1])
    red_ts = ots[red_idx] + np.timedelta64(5, "s")
    w.tables["promotion_redemptions"] = pd.DataFrame(
        {
            "redemption_id": np.arange(1, len(red_idx) + 1),
            "promo_id": promos["promo_id"].to_numpy()[promo_of[red_idx]],
            "order_id": red_idx + 1,
            "customer_id": o["customer_id"].to_numpy()[red_idx],
            "discount_bdt": np.round(disc_by_order[red_idx], 2),
            "redeemed_at": red_ts,
        }
    )


def build_status_events(w: World) -> None:
    """Event sequence per order (UTC timestamps). Exact row total via optional 'confirmed' events."""
    o = w.tables["orders"]
    n = w.n("order_status_events")
    rng = w.rng("order_status_events", "times")
    st = o["status"].to_numpy()
    placed_utc = o["order_at"].to_numpy() - w.utc_offset
    ship = w.meta["first_ship_utc"]  # from logistics: first shipment time per order (NaT if none)
    deliv = w.meta["last_delivered_utc"]  # last delivery per order (NaT if none)
    ret_at = w.meta["return_requested_utc"]  # first return request per order (NaT if none)
    seqs: list[list[tuple[str, np.datetime64, str]]] = []
    for i in range(len(o)):
        s = st[i]
        t0 = placed_utc[i]
        seq = [("placed", t0, "customer")]
        if s in ("cancelled", "canceled"):
            seq.append((s, t0 + np.timedelta64(int(rng.integers(600, 2 * 86400)), "s"), "customer"))
        elif s == "payment_failed":
            seq.append(("payment_failed", t0 + np.timedelta64(int(rng.integers(30, 1800)), "s"), "system"))
        else:
            seq.append(("shipped", ship[i], "seller"))
            if s in ("delivered", "returned"):
                seq.append(("delivered", deliv[i], "courier"))
            if s == "returned":
                seq.append(("returned", ret_at[i], "customer"))
        seqs.append(seq)
    base = sum(len(s) for s in seqs)
    extra = n - base
    eligible = np.flatnonzero(np.isin(st, SHIPPED_STATUSES))
    if extra < 0 or extra > len(eligible):
        raise RuntimeError(f"cannot reach {n} status events from base {base}")
    conf = set(rng.choice(eligible, size=extra, replace=False).tolist())
    rows_o, rows_s, rows_t, rows_a = [], [], [], []
    for i, seq in enumerate(seqs):
        if i in conf:
            t0 = seq[0][1]
            t1 = seq[1][1]
            gap = max(int((t1 - t0) / np.timedelta64(1, "s")), 120)
            seq.insert(1, ("confirmed", t0 + np.timedelta64(int(rng.integers(60, gap)), "s"), "seller"))
        for s, t, a in seq:
            rows_o.append(i + 1)
            rows_s.append(s)
            rows_t.append(t)
            rows_a.append(a)
    df = pd.DataFrame(
        {
            "order_id": rows_o,
            "status": rows_s,
            "event_at_utc": np.array(rows_t, dtype="datetime64[s]"),
            "actor": rows_a,
        }
    )
    df.insert(0, "event_id", np.arange(1, len(df) + 1))
    w.tables["order_status_events"] = df


def gmv_top_day(w: World) -> tuple[np.datetime64, float]:
    """(busiest day by GMV of valid orders, ratio to the runner-up)."""
    o = w.tables["orders"]
    valid = ~o["is_deleted"].to_numpy() & ~np.isin(o["status"].to_numpy(), VALID_EXCLUDED)
    day = o["order_at"].to_numpy().astype("datetime64[D]")[valid]
    val = w.meta["order_value_bdt"][valid]
    s = pd.Series(val).groupby(day).sum().sort_values(ascending=False)
    return np.datetime64(s.index[0], "D"), float(s.iloc[0] / s.iloc[1])


def build_orders_and_items(w: World) -> None:
    """Orders + items; F05 raises the 11.11 2025 volume margin until that day is the top GMV day."""
    base = float(w.params["orders"]["sale_1111_2025_margin"])
    for margin in (base, base * 1.2, base * 1.45, base * 1.75, base * 2.1):
        build_orders(w, margin)
        build_items(w)
        top, ratio = gmv_top_day(w)
        if top == np.datetime64("2025-11-11") and ratio >= 1.05:
            w.meta["f05_margin"] = margin
            return
    raise RuntimeError("F05 could not be planted")
