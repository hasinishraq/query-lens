"""Payments (with failed retries), returns and refunds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import DAY, World, probs
from ..rng import exact_choice

F07_DAYS = ("2025-08-12", "2025-08-13", "2025-08-14")
F07_TARGET = 0.81
F02_TARGET = 0.094


def build_payments(w: World) -> None:
    pp = w.params["payments"]
    o = w.tables["orders"]
    n_o = len(o)
    rng = w.rng("payments", "attrs")
    mk, mp = probs(pp["method"])
    method = exact_choice(w.rng("payments", "method"), mk, mp, n_o).astype(object)
    usd = o["currency"].to_numpy() == "USD"
    uk, up = probs(pp["usd_methods"])
    method[usd] = rng.choice(np.array(uk, dtype=object), size=int(usd.sum()), p=np.array(up) / sum(up))
    st = o["status"].to_numpy()
    cancelled = np.isin(st, ["cancelled", "canceled"])
    pf = st == "payment_failed"
    fk, fp = probs(pp["failed_order_method"])
    sw = pf & (method == "cod")  # COD orders cannot fail payment: they paid online and failed
    method[sw] = rng.choice(np.array(fk, dtype=object), size=int(sw.sum()), p=np.array(fp) / sum(fp))
    no_pay_extra = (pp["cancelled_without_payment_share"] - mp[mk.index("cod")]) / (1 - mp[mk.index("cod")])
    no_payment = cancelled & ((method == "cod") | (rng.random(n_o) < no_pay_extra))
    has_pay = ~no_payment
    value = w.meta["order_value_bdt"]
    order_utc = o["order_at"].to_numpy() - w.utc_offset
    deliv = w.meta["last_delivered_utc"]

    final_status = np.empty(n_o, dtype=object)
    final_status[:] = "captured"
    final_status[pf] = "failed"
    final_status[cancelled & has_pay] = "voided"
    cod = method == "cod"
    final_status[cod & (st == "in_transit")] = "pending"

    base = int(has_pay.sum())
    n_target = w.n("payments")
    n_retry = n_target - base
    retries = np.zeros(n_o, dtype=np.int64)
    # F07: bKash success drops to 81% on three August 2025 days
    oday = o["order_at"].to_numpy().astype("datetime64[D]")
    f07 = np.isin(oday, np.array(F07_DAYS, dtype="datetime64[D]")) & (method == "bkash") & has_pay
    cap_f07 = f07 & (final_status == "captured")
    fail_f07 = f07 & (final_status == "failed")
    c = int(cap_f07.sum())
    need_failed = int(np.ceil(c * (1 - F07_TARGET) / F07_TARGET)) - int(fail_f07.sum())
    f07_idx = np.flatnonzero(cap_f07)
    if need_failed > 0 and len(f07_idx):
        per = np.zeros(len(f07_idx), dtype=np.int64)
        for j in range(need_failed):
            per[j % len(f07_idx)] += 1
        retries[f07_idx] = per
    remaining = n_retry - int(retries.sum())
    retry_ok = has_pay & (final_status != "failed") & ~f07 & ~(cod & (final_status == "pending"))
    cand = np.flatnonzero(retry_ok & (method != "cod"))
    n_orders_retry = min(len(cand), round(n_o * pp["retry_order_share"]))
    rw = np.array([pp["retry_weight_by_method"][m] for m in method[cand]], dtype=float)
    chosen = w.rng("payments", "retry").choice(cand, size=n_orders_retry, replace=False, p=rw / rw.sum())
    retries[chosen] += 1
    remaining -= n_orders_retry
    if remaining > 0:
        more = w.rng("payments", "retry2").choice(chosen, size=remaining, replace=remaining > len(chosen))
        np.add.at(retries, more, 1)
    elif remaining < 0:
        drop = chosen[:-remaining]
        retries[drop] -= 1

    prov = {m: probs(d) for m, d in pp["providers"].items()}
    rows = []
    for i in np.flatnonzero(has_pay):
        m = method[i]
        pk, pv = prov[m]
        provider = pk[int(rng.choice(len(pk), p=np.array(pv) / sum(pv)))]
        t = order_utc[i] + np.timedelta64(int(rng.integers(20, 300)), "s")
        for a in range(int(retries[i])):
            rows.append((i + 1, m, provider, a + 1, "failed", value[i], t, np.datetime64("NaT", "s")))
            t = t + np.timedelta64(int(rng.integers(30, 900)), "s")
        fs = final_status[i]
        if fs == "captured":
            cap = deliv[i] if m == "cod" else t + np.timedelta64(int(rng.integers(2, 60)), "s")
        else:
            cap = np.datetime64("NaT", "s")
        rows.append((i + 1, m, provider, int(retries[i]) + 1, fs, value[i], t, cap))
    df = pd.DataFrame(
        rows,
        columns=[
            "order_id",
            "method",
            "provider",
            "attempt_no",
            "status",
            "amount_bdt",
            "created_at_utc",
            "captured_at_utc",
        ],
    )
    df["created_at_utc"] = df["created_at_utc"].astype("datetime64[s]")
    df["captured_at_utc"] = df["captured_at_utc"].astype("datetime64[s]")
    df.insert(0, "payment_id", np.arange(1, len(df) + 1))
    if len(df) != n_target:
        raise RuntimeError(f"{len(df)} payments, expected {n_target}")
    w.tables["payments"] = df


def build_returns_refunds(w: World) -> None:
    rp = w.params["refunds"]
    o = w.tables["orders"]
    items = w.tables["order_items"]
    pay = w.tables["payments"]
    rng = w.rng("returns", "attrs")
    rng_r = w.rng("refunds", "attrs")
    starts = w.meta["item_starts"]
    st = o["status"].to_numpy()
    n_o = len(o)
    deliv_local = w.meta["last_delivered_utc"] + w.utc_offset
    end_ts = (w.end + DAY).astype("datetime64[s]") - np.timedelta64(1, "s")

    returned = np.flatnonzero(st == "returned")
    k_per = np.diff(starts)
    n_ret = w.n("returns")
    per = np.ones(len(returned), dtype=np.int64)
    room = k_per[returned] - 1
    extra = n_ret - len(returned)
    if extra < 0 or extra > room.sum():
        raise RuntimeError("cannot place returns")
    slots = np.repeat(np.arange(len(returned)), room)
    pick = rng.choice(len(slots), size=extra, replace=False)
    np.add.at(per, slots[pick], 1)
    ret_item, ret_order = [], []
    for j, oi in enumerate(returned):
        a = starts[oi]
        for k in range(int(per[j])):
            ret_item.append(a + k)
            ret_order.append(oi)
    ret_item_a = np.array(ret_item)
    ret_order_a = np.array(ret_order)
    nr = len(ret_item_a)
    sk, sp = probs(rp["return_status"])
    r_status = exact_choice(rng, sk, sp, nr).astype(object)
    rk, rpp = probs(rp["return_reasons"])
    reason = exact_choice(rng, rk, rpp, nr)
    req = deliv_local[ret_order_a] + (rng.integers(1, 10, nr) * 86400 + rng.integers(0, 36000, nr)).astype(
        "timedelta64[s]"
    )
    req = np.minimum(req, end_ts - np.timedelta64(3 * 86400, "s"))
    req = np.maximum(req, deliv_local[ret_order_a] + np.timedelta64(600, "s"))
    recv = req + (rng.integers(2, 7, nr) * 86400).astype("timedelta64[s]")
    late_recv = recv > end_ts
    r_status[late_recv & (r_status == "refunded")] = "requested"
    recv = np.where(r_status == "requested", np.datetime64("NaT", "s"), recv)
    qty_item = items["quantity"].to_numpy()[ret_item_a]
    r_qty = np.where(qty_item > 1, rng.integers(1, qty_item + 1), 1)

    captured = (
        pay[pay["status"] == "captured"].drop_duplicates("order_id").set_index("order_id")["payment_id"]
    )
    cap_pay = np.full(n_o, -1, dtype=np.int64)
    cap_pay[captured.index.to_numpy() - 1] = captured.to_numpy()

    line = items["line_total_bdt"].to_numpy()
    refunds = []  # (order_idx, payment_id, amount, reason, requested, processed)
    ret_refund_idx = np.full(nr, -1, dtype=np.int64)
    for j in range(nr):
        if r_status[j] != "refunded":
            continue
        oi = ret_order_a[j]
        amt = round(float(line[ret_item_a[j]]) * r_qty[j] / qty_item[j], 2)
        proc = recv[j] + np.timedelta64(int(rng_r.integers(86400, 5 * 86400)), "s")
        proc = min(proc, end_ts)
        ret_refund_idx[j] = len(refunds)
        refunds.append((oi, cap_pay[oi], amt, f"return:{reason[j]}", recv[j], proc))

    # goodwill refunds on delivered orders; F02 cell (Chattogram, March 2026) is planted to 9.4%
    n_ref = w.n("refunds")
    n_goodwill = n_ref - len(refunds)
    gk, gp = probs(rp["goodwill_reasons"])
    lo, hi = rp["goodwill_share_of_total"]
    oday = o["order_at"].to_numpy().astype("datetime64[D]")
    deleted = o["is_deleted"].to_numpy()
    addr = w.tables["customer_addresses"].set_index("address_id")
    city_region = w.tables["cities"].set_index("city_id")["region_id"]
    region = city_region.loc[addr.loc[o["address_id"].to_numpy(), "city_id"].to_numpy()].to_numpy()
    cell = (region == 2) & (oday >= np.datetime64("2026-03-01")) & (oday <= np.datetime64("2026-03-31"))
    has_cap = cap_pay >= 0
    refunded_orders = {r[0] for r in refunds}
    cell_den = np.flatnonzero(cell & has_cap & ~deleted)
    cell_have = sum(1 for i in cell_den if i in refunded_orders)
    cell_need = max(0, round(F02_TARGET * len(cell_den)) - cell_have)
    ship = w.tables["shipments"]
    fastbox_id = int(w.tables["couriers"].set_index("name").loc["FastBox", "courier_id"])
    fb_orders = set((ship.loc[ship["courier_id"] == fastbox_id, "order_id"] - 1).tolist())
    cell_cand = [i for i in cell_den if st[i] == "delivered" and i not in refunded_orders]
    cell_cand.sort(key=lambda i: (i not in fb_orders, i))
    cell_pick = cell_cand[:cell_need]
    value = w.meta["order_value_bdt"]

    def goodwill(oi: int, reason_txt: str) -> None:
        amt = round(float(value[oi]) * float(rng_r.uniform(lo, hi)), 2)
        reqt = deliv_local[oi] + np.timedelta64(int(rng_r.integers(3600, 7 * 86400)), "s")
        reqt = min(reqt, end_ts - np.timedelta64(86400, "s"))
        proc = reqt + np.timedelta64(int(rng_r.integers(3600, 4 * 86400)), "s")
        proc = min(proc, end_ts)
        refunds.append((oi, cap_pay[oi], amt, reason_txt, reqt, proc))
        refunded_orders.add(oi)

    for oi in cell_pick:
        goodwill(int(oi), "damaged_in_transit")
    rest = n_goodwill - len(cell_pick)
    pool = np.flatnonzero((st == "delivered") & has_cap & ~cell)
    pool = np.array([i for i in pool if i not in refunded_orders])
    gpick = rng_r.choice(pool, size=rest, replace=False)
    greason = exact_choice(rng_r, gk, gp, rest)
    for oi, rs in zip(gpick, greason, strict=True):
        goodwill(int(oi), str(rs))

    rdf = pd.DataFrame(
        refunds, columns=["order_idx", "payment_id", "amount_bdt", "reason", "requested_at", "processed_at"]
    )
    rdf["order_id"] = rdf["order_idx"] + 1
    rdf = rdf.sort_values(["requested_at", "order_id"], kind="stable")
    new_id = np.empty(len(rdf), dtype=np.int64)
    new_id[rdf.index.to_numpy()] = np.arange(1, len(rdf) + 1)
    rdf = rdf.reset_index(drop=True)
    rdf.insert(0, "refund_id", np.arange(1, len(rdf) + 1))
    rdf["requested_at"] = rdf["requested_at"].astype("datetime64[s]")
    rdf["processed_at"] = rdf["processed_at"].astype("datetime64[s]")
    w.tables["refunds"] = rdf[
        ["refund_id", "order_id", "payment_id", "amount_bdt", "reason", "requested_at", "processed_at"]
    ]
    refund_id = np.where(ret_refund_idx >= 0, new_id[np.maximum(ret_refund_idx, 0)], 0)
    ret = pd.DataFrame(
        {
            "return_id": np.arange(1, nr + 1),
            "order_item_id": ret_item_a + 1,
            "quantity": r_qty,
            "reason": reason,
            "status": r_status,
            "requested_at": req,
            "received_at": recv.astype("datetime64[s]"),
            "refund_id": pd.array(refund_id, dtype="Int64"),
        }
    )
    ret.loc[ret_refund_idx < 0, "refund_id"] = pd.NA
    w.tables["returns"] = ret
    ret_req_utc = np.full(n_o, np.datetime64("NaT", "s"), dtype="datetime64[s]")
    g = pd.DataFrame({"o": ret_order_a, "t": req - w.utc_offset}).groupby("o")["t"].min()
    ret_req_utc[g.index.to_numpy()] = g.to_numpy()
    w.meta["return_requested_utc"] = ret_req_utc
