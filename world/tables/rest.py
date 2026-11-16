"""Tables built after the order flow: loyalty, inventory, web sessions, support, reviews."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import DAY, World, probs
from ..pools import REVIEW_BODIES, REVIEW_TITLES, SUPPORT_AGENT, SUPPORT_BOT, SUPPORT_CUSTOMER
from ..rng import allocate, exact_choice
from .logistics import BARISHAL_CLOSED

VALID_EXCLUDED = ("cancelled", "canceled", "payment_failed")


def _end_ts(w: World) -> np.datetime64:
    return (w.end + DAY).astype("datetime64[s]") - np.timedelta64(1, "s")


# --------------------------------------------------------------------------- loyalty
def build_loyalty(w: World) -> None:
    rng = w.rng("loyalty_accounts", "attrs")
    cust = w.tables["customers"]
    o = w.tables["orders"]
    n_acc = w.n("loyalty_accounts")
    n_txn = w.n("loyalty_transactions")
    n_c = len(cust)
    counts = np.bincount(o["customer_id"].to_numpy() - 1, minlength=n_c).astype(float)
    wts = (counts + 0.5) * (~cust["is_deleted"].to_numpy())
    members = np.sort(rng.choice(n_c, size=n_acc, replace=False, p=wts / wts.sum()))
    signup = cust["signup_at"].to_numpy()[members]
    end_ts = _end_ts(w)
    span = ((end_ts - np.timedelta64(30 * 86400, "s")) - signup) / np.timedelta64(1, "s")
    enrolled = signup + (rng.random(n_acc) * np.maximum(span, 0) * 0.5).astype("timedelta64[s]")
    loyalty_id = np.arange(1, n_acc + 1)
    member_of = np.full(n_c, -1, dtype=np.int64)
    member_of[members] = np.arange(n_acc)

    st = o["status"].to_numpy()
    good = np.isin(st, ["delivered", "returned"]) & ~o["is_deleted"].to_numpy()
    oc = o["customer_id"].to_numpy() - 1
    ots = o["order_at"].to_numpy()
    acc = member_of[oc]
    earn_ok = good & (acc >= 0)
    earn_ok &= ots >= enrolled[np.maximum(acc, 0)]
    earn_idx = np.flatnonzero(earn_ok)
    n_earn = min(len(earn_idx), round(n_txn * 0.8))
    rng_t = w.rng("loyalty_transactions", "attrs")
    earn_idx = np.sort(rng_t.choice(earn_idx, size=n_earn, replace=False))
    value = w.meta["order_value_bdt"]
    n_other = n_txn - n_earn
    n_redeem = round(n_other * 0.75)
    n_expire = n_other - n_redeem
    rows = []  # (acc, order_id|None, type, points, ts, expires)
    for i in earn_idx:
        pts = max(1, int(value[i] // 100))
        t = ots[i] + np.timedelta64(int(rng_t.integers(86400, 10 * 86400)), "s")
        t = min(t, end_ts)
        rows.append((int(acc[i]), int(i + 1), "earn", pts, t, t + np.timedelta64(365 * 86400, "s")))
    # redeem at checkout of a later order of the same member; expire unused points
    by_acc: dict[int, list[int]] = {}
    for i in earn_idx:
        by_acc.setdefault(int(acc[i]), []).append(int(i))
    accs = np.array(sorted(by_acc))
    red_acc = rng_t.choice(accs, size=n_redeem, replace=True)
    for a in red_acc:
        oi = by_acc[int(a)][int(rng_t.integers(0, len(by_acc[int(a)])))]
        rows.append(
            (
                int(a),
                oi + 1,
                "redeem",
                int(rng_t.choice([50, 100, 200, 300])),
                ots[oi] + np.timedelta64(60, "s"),
                None,
            )
        )
    exp_acc = rng_t.choice(accs, size=n_expire, replace=True)
    for a in exp_acc:
        first = by_acc[int(a)][0]
        t = min(ots[first] + np.timedelta64(int(rng_t.integers(366, 700)) * 86400, "s"), end_ts)
        rows.append((int(a), None, "expire", int(rng_t.choice([10, 25, 50, 80])), t, None))
    df = pd.DataFrame(rows, columns=["acc", "order_id", "txn_type", "points", "txn_at", "expires_at"])
    df["txn_at"] = df["txn_at"].astype("datetime64[s]")
    df["expires_at"] = df["expires_at"].astype("datetime64[s]")
    df = df.sort_values(["txn_at", "acc"], kind="stable").reset_index(drop=True)
    # cap redeem/expire so no balance goes negative
    bal = np.zeros(n_acc, dtype=np.int64)
    pts = df["points"].to_numpy().copy()
    ttype = df["txn_type"].to_numpy()
    accs_col = df["acc"].to_numpy()
    for j in range(len(df)):
        a = accs_col[j]
        if ttype[j] == "earn":
            bal[a] += pts[j]
        else:
            pts[j] = max(1, min(pts[j], bal[a])) if bal[a] > 0 else 1
            bal[a] = max(0, bal[a] - pts[j])
    df["points"] = pts
    df.insert(0, "loyalty_txn_id", np.arange(1, len(df) + 1))
    df["loyalty_id"] = loyalty_id[df["acc"].to_numpy()]
    df["order_id"] = df["order_id"].astype("Int64")
    w.tables["loyalty_transactions"] = df[
        ["loyalty_txn_id", "loyalty_id", "order_id", "txn_type", "points", "txn_at", "expires_at"]
    ]
    signed = np.where(ttype == "earn", pts, -pts)
    balance = np.bincount(accs_col, weights=signed, minlength=n_acc)
    balance = np.maximum(balance, 0).astype(np.int64)
    tier = np.where(
        balance >= 3000,
        "platinum",
        np.where(balance >= 1200, "gold", np.where(balance >= 400, "silver", "basic")),
    )
    w.tables["loyalty_accounts"] = pd.DataFrame(
        {
            "loyalty_id": loyalty_id,
            "customer_id": members + 1,
            "tier": tier,
            "points_balance": balance,
            "enrolled_at": enrolled.astype("datetime64[s]"),
            "status": np.where(rng.random(n_acc) < 0.9, "active", "inactive"),
        }
    )


# --------------------------------------------------------------------------- inventory
def build_inventory(w: World) -> None:
    ip = w.params["inventory"]
    rng = w.rng("inventory_snapshots", "attrs")
    wh = w.tables["warehouses"]
    n = w.n("inventory_snapshots")
    first_sun = w.start + ((6 - pd.Timestamp(str(w.start)).day_of_week) % 7) * DAY
    weeks = np.arange(first_sun, w.end + DAY, 7 * DAY)
    closed = np.datetime64(BARISHAL_CLOSED, "D")
    slots = []  # (warehouse_idx, week_date)
    for i, r in enumerate(wh.itertuples(index=False)):
        for d in weeks:
            if d < np.datetime64(r.opened_at, "D"):
                continue
            if r.name == "Barishal-1" and d > closed:
                continue
            slots.append((i, d))
    per = allocate(n, np.ones(len(slots)))
    max_per = int(per.max())
    n_var = len(w.tables["product_variants"])
    tracked = [rng.choice(n_var, size=max_per, replace=False) + 1 for _ in range(len(wh))]
    gaz2 = int(np.flatnonzero(wh["name"].to_numpy() == "Gazipur-2")[0])
    d_l, w_l, v_l = [], [], []
    for (wi, d), k in zip(slots, per, strict=True):
        d_l.extend([d] * int(k))
        w_l.extend([wi + 1] * int(k))
        v_l.extend(tracked[wi][: int(k)].tolist())
    d_a = np.array(d_l, dtype="datetime64[D]")
    w_a = np.array(w_l)
    rows = len(d_a)
    p_out = np.full(rows, ip["stockout_base"])
    p_out[(w_a == gaz2 + 1) & (d_a >= np.datetime64("2026-01-01"))] = ip["stockout_gazipur2_2026"]
    out = rng.random(rows) < p_out
    on_hand = np.where(out, 0, rng.poisson(45, rows) + 1)
    reserved = np.where(out, 0, np.minimum(on_hand, rng.poisson(4, rows)))
    w.tables["inventory_snapshots"] = pd.DataFrame(
        {
            "snapshot_date": d_a,
            "warehouse_id": w_a,
            "variant_id": np.array(v_l),
            "on_hand_units": on_hand,
            "reserved_units": reserved,
            "reorder_point": rng.choice([5, 10, 15, 20], rows),
        }
    )


# --------------------------------------------------------------------------- web
def build_web_sessions(w: World) -> None:
    wp = w.params["web"]
    rng = w.rng("web_sessions", "attrs")
    o = w.tables["orders"]
    cust = w.tables["customers"]
    n = w.n("web_sessions")
    conv = np.flatnonzero(o["channel"].to_numpy() == "web")
    n_conv = len(conv)
    n_other = n - n_conv
    order_utc = o["order_at"].to_numpy()[conv] - w.utc_offset
    c_start = order_utc - (rng.integers(120, 3600, n_conv)).astype("timedelta64[s]")
    c_cust = o["customer_id"].to_numpy()[conv]
    c_dev = np.where(o["device_type"].to_numpy()[conv] == "desktop", "desktop", "mobile_web")
    # non-converting sessions follow the day weights; 40% anonymous
    dw = w.meta["day_weight"]
    day = rng.choice(len(dw), size=n_other, p=dw / dw.sum())
    ts = (
        w.days[day].astype("datetime64[s]")
        + (rng.integers(0, 86400, n_other)).astype("timedelta64[s]")
        - w.utc_offset
    )
    anon = rng.random(n_other) < wp["anonymous_share"]
    signup_utc = cust["signup_at"].to_numpy() - w.utc_offset
    cand = rng.integers(0, len(cust), n_other)
    ok = signup_utc[cand] <= ts
    anon |= ~ok
    o_cust = np.where(anon, -1, cand + 1)
    start = np.concatenate([c_start, ts])
    custs = np.concatenate([c_cust, o_cust])
    conv_order = np.concatenate([conv + 1, np.full(n_other, -1)])
    dev = np.concatenate([c_dev, rng.choice(["desktop", "mobile_web"], n_other, p=[0.4, 0.6])])
    tk, tp = probs(wp["traffic_source"])
    src = exact_choice(rng, tk, tp, n)
    camp = w.tables["marketing_campaigns"]
    cs = camp["start_date"].to_numpy().astype("datetime64[D]")
    ce = camp["end_date"].to_numpy().astype("datetime64[D]")
    cid = camp["campaign_id"].to_numpy()
    local_day = (start + w.utc_offset).astype("datetime64[D]")
    campaign = np.full(n, -1, dtype=np.int64)
    paid = np.isin(src, ["paid_search", "social", "email"])
    for i in np.flatnonzero(paid):
        act = np.flatnonzero((cs <= local_day[i]) & (ce >= local_day[i]))
        if len(act):
            campaign[i] = cid[act[int(rng.integers(0, len(act)))]]
    pages = np.maximum(1, np.round(rng.lognormal(1.3, 0.7, n))).astype(int)
    is_conv = conv_order > 0
    pages = np.where(is_conv, np.maximum(pages, 3), pages)
    dur = (pages * rng.integers(20, 120, n)).astype("timedelta64[s]")
    landing = rng.choice(
        ["/", "/deals", "/category", "/product", "/search", "/campaign"],
        n,
        p=[0.3, 0.12, 0.2, 0.25, 0.08, 0.05],
    )
    added = is_conv | (rng.random(n) < 0.18)
    order = np.argsort(start, kind="stable")
    end_utc = _end_ts(w) - w.utc_offset
    df = pd.DataFrame(
        {
            "session_id": np.arange(1, n + 1),
            "customer_id": pd.array(custs[order], dtype="Int64"),
            "started_at_utc": start[order],
            "ended_at_utc": np.minimum(start[order] + dur[order], end_utc),
            "device_type": dev[order],
            "traffic_source": src[order],
            "campaign_id": pd.array(campaign[order], dtype="Int64"),
            "landing_page": landing[order],
            "pages_viewed": pages[order],
            "added_to_cart": added[order],
            "converted_order_id": pd.array(conv_order[order], dtype="Int64"),
        }
    )
    for c in ("customer_id", "campaign_id", "converted_order_id"):
        df.loc[df[c] < 0, c] = pd.NA
    w.tables["web_sessions"] = df


# --------------------------------------------------------------------------- support
CATS = {"delivery": 0.3, "refund": 0.2, "payment": 0.15, "product": 0.25, "account": 0.1}


def _lang(text: str) -> str:
    if any("ঀ" <= ch <= "৿" for ch in text):
        return "bn"
    banglish = (
        "ekhono",
        "kobe",
        "hoye",
        "pathiyeche",
        "ashche",
        "apnar",
        "korun",
        "valo",
        "koro",
        "ager",
        "dekhao",
        "ache",
        "nosto",
        "onujayi",
        "ekta",
        "na.",
        "amar",
        "ektu",
    )
    return "banglish" if any(b in text.lower() for b in banglish) else "en"


def build_support(w: World) -> None:
    rng = w.rng("support_tickets", "attrs")
    o = w.tables["orders"]
    n = w.n("support_tickets")
    end_utc = _end_ts(w) - w.utc_offset
    ck, cp = probs(CATS)
    cat = exact_choice(rng, ck, cp, n)
    with_order = rng.random(n) < 0.82
    oi = rng.integers(0, len(o), n)
    st = o["status"].to_numpy()
    # payment tickets lean towards failed-payment orders
    pf = np.flatnonzero(st == "payment_failed")
    pay_t = (cat == "payment") & with_order & (rng.random(n) < 0.5)
    oi[pay_t] = rng.choice(pf, size=int(pay_t.sum()))
    cust = o["customer_id"].to_numpy()[oi]
    any_cust = rng.integers(1, len(w.tables["customers"]) + 1, n)
    cust = np.where(with_order, cust, any_cust)
    base_t = np.where(
        with_order,
        o["order_at"].to_numpy()[oi] - w.utc_offset,
        w.tables["customers"]["signup_at"].to_numpy()[any_cust - 1] - w.utc_offset,
    )
    created = base_t + (rng.integers(3600, 12 * 86400, n)).astype("timedelta64[s]")
    created = np.minimum(created, end_utc - np.timedelta64(3600, "s"))
    status = exact_choice(rng, ["resolved", "closed", "open", "pending"], [0.8, 0.1, 0.06, 0.04], n)
    hours = np.round(rng.lognormal(2.6, 1.0, n) * 3600).astype("timedelta64[s]")
    resolved = created + hours
    is_res = np.isin(status, ["resolved", "closed"]) & (resolved <= end_utc)
    status = np.where(np.isin(status, ["resolved", "closed"]) & ~is_res, "open", status)
    order = np.argsort(created, kind="stable")
    t = pd.DataFrame(
        {
            "ticket_id": np.arange(1, n + 1),
            "customer_id": cust[order],
            "order_id": pd.array(np.where(with_order, oi + 1, -1)[order], dtype="Int64"),
            "channel": rng.choice(["chat", "phone", "email", "facebook"], n, p=[0.45, 0.25, 0.15, 0.15])[
                order
            ],
            "category": cat[order],
            "priority": rng.choice(["low", "medium", "high", "urgent"], n, p=[0.3, 0.45, 0.2, 0.05])[order],
            "status": status[order],
            "created_at_utc": created[order],
            "resolved_at_utc": np.where(is_res, resolved, np.datetime64("NaT", "s"))[order],
            "agent_id": rng.integers(1, 41, n)[order],
        }
    )
    t.loc[t["order_id"] < 0, "order_id"] = pd.NA
    w.tables["support_tickets"] = t

    rng_m = w.rng("support_messages", "attrs")
    n_m = w.n("support_messages")
    per = allocate(n_m - 2 * n, rng_m.gamma(1.5, 1.0, n) + 0.1) + 2
    rows = []
    tc = t["category"].to_numpy()
    tcr = t["created_at_utc"].to_numpy()
    tre = t["resolved_at_utc"].to_numpy()
    for i in range(n):
        k = int(per[i])
        t_end = tre[i] if not np.isnat(tre[i]) else min(tcr[i] + np.timedelta64(3 * 86400, "s"), end_utc)
        span = max(int((t_end - tcr[i]) / np.timedelta64(1, "s")), k * 60)
        offs = np.sort(rng_m.integers(0, span, k))
        offs[0] = 0
        for j in range(k):
            if j == 0:
                sender = "customer"
                body = SUPPORT_CUSTOMER[tc[i]][int(rng_m.integers(0, 4))]
            elif j == 1 and rng_m.random() < 0.3:
                sender = "bot"
                body = SUPPORT_BOT[int(rng_m.integers(0, len(SUPPORT_BOT)))]
            elif j % 2 == 1:
                sender = "agent"
                body = SUPPORT_AGENT[int(rng_m.integers(0, len(SUPPORT_AGENT)))]
            else:
                sender = "customer"
                body = SUPPORT_CUSTOMER[tc[i]][int(rng_m.integers(0, 4))]
            rows.append(
                (i + 1, sender, min(tcr[i] + np.timedelta64(int(offs[j]), "s"), end_utc), body, _lang(body))
            )
    m = pd.DataFrame(rows, columns=["ticket_id", "sender_type", "sent_at_utc", "body", "language"])
    m["sent_at_utc"] = m["sent_at_utc"].astype("datetime64[s]")
    m.insert(0, "message_id", np.arange(1, len(m) + 1))
    w.tables["support_messages"] = m


def build_reviews(w: World) -> None:
    rp = w.params["reviews"]
    rng = w.rng("product_reviews", "attrs")
    o = w.tables["orders"]
    items = w.tables["order_items"]
    var = w.tables["product_variants"].set_index("variant_id")["product_id"]
    n = w.n("product_reviews")
    starts = w.meta["item_starts"]
    k_per = np.diff(starts)
    st = o["status"].to_numpy()
    dlv = np.flatnonzero((st == "delivered") & ~o["is_deleted"].to_numpy())
    n_orders = round(len(dlv) * rp["delivered_order_review_share"])
    n_orders = min(n_orders, n)
    rev_orders = np.sort(rng.choice(dlv, size=n_orders, replace=False))
    room = k_per[rev_orders] - 1
    extra = n - n_orders
    if extra > room.sum():
        raise RuntimeError("not enough items for reviews")
    slots = np.repeat(np.arange(n_orders), room)
    pick = rng.choice(len(slots), size=extra, replace=False)
    per = np.ones(n_orders, dtype=np.int64)
    np.add.at(per, slots[pick], 1)
    item_idx = np.concatenate([starts[oi] + np.arange(per[j]) for j, oi in enumerate(rev_orders)])
    order_idx = np.repeat(rev_orders, per)
    rk, rpp = probs({int(k): v for k, v in rp["rating"].items()})
    rating = exact_choice(rng, rk, rpp, n).astype(int)
    deliv_local = w.meta["last_delivered_utc"][order_idx] + w.utc_offset
    end_ts = _end_ts(w)
    created = np.minimum(deliv_local + (rng.integers(3600, 20 * 86400, n)).astype("timedelta64[s]"), end_ts)
    title = [REVIEW_TITLES[r][int(rng.integers(0, 5))] for r in rating]
    body = [REVIEW_BODIES[r][int(rng.integers(0, 5))] for r in rating]
    order = np.argsort(created, kind="stable")
    w.tables["product_reviews"] = pd.DataFrame(
        {
            "review_id": np.arange(1, n + 1),
            "product_id": var.loc[items["variant_id"].to_numpy()[item_idx]].to_numpy()[order],
            "customer_id": o["customer_id"].to_numpy()[order_idx][order],
            "order_id": (order_idx + 1)[order],
            "rating": rating[order],
            "title": np.array(title, dtype=object)[order],
            "body": np.array(body, dtype=object)[order],
            "created_at": created[order],
        }
    )
