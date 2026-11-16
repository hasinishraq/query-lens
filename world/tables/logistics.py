"""Logistics: couriers and warehouses (fixed), shipments and shipment events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import DAY, World
from ..pools import SHIPMENT_NOTES
from .orders import SHIPPED_STATUSES
from .sellers import PADMA

FASTBOX = "FastBox"
SERVICE = {
    "FastBox": "express",
    "Pathao Courier": "express",
    "RedX": "standard",
    "Steadfast": "standard",
    "Paperfly": "standard",
    "eCourier": "express",
    "Sundarban Courier": "economy",
    "SA Paribahan": "economy",
    "Delivery Tiger": "standard",
    "Bazaar Express": "express",
    "Bazaar Express Plus": "same_day",
    "Janani Express": "economy",
}

# (name, district, capacity, opened, active)
WAREHOUSES = [
    ("Dhaka-Tejgaon", "Dhaka", 120_000, "2021-03-01", True),
    ("Gazipur-1", "Gazipur", 150_000, "2021-09-15", True),
    ("Gazipur-2", "Gazipur", 90_000, "2023-02-01", True),
    ("Narayanganj-1", "Narayanganj", 80_000, "2022-05-10", True),
    ("Chattogram-Port", "Chattogram", 110_000, "2021-11-20", True),
    ("Cumilla-1", "Cumilla", 40_000, "2023-08-01", True),
    ("Khulna-1", "Khulna", 45_000, "2022-07-01", True),
    ("Rajshahi-1", "Rajshahi", 40_000, "2022-10-01", True),
    ("Bogura-1", "Bogura", 35_000, "2025-03-01", True),
    ("Sylhet-1", "Sylhet", 35_000, "2023-01-15", True),
    ("Rangpur-1", "Rangpur", 30_000, "2024-04-01", True),
    ("Barishal-1", "Barishal", 25_000, "2022-12-01", False),
]
BARISHAL_CLOSED = "2026-01-31"


def build_couriers(w: World) -> None:
    names = list(w.params["logistics"]["courier_late_rate"])
    w.tables["couriers"] = pd.DataFrame(
        {
            "courier_id": np.arange(1, len(names) + 1),
            "name": names,
            "service_level": [SERVICE[nm] for nm in names],
            "is_third_party": [not nm.startswith("Bazaar Express") for nm in names],
        }
    )


def build_warehouses(w: World) -> None:
    cities = w.tables["cities"].set_index("name")["city_id"]
    w.tables["warehouses"] = pd.DataFrame(
        {
            "warehouse_id": np.arange(1, len(WAREHOUSES) + 1),
            "name": [x[0] for x in WAREHOUSES],
            "city_id": [int(cities[x[1]]) for x in WAREHOUSES],
            "capacity_units": [x[2] for x in WAREHOUSES],
            "opened_at": [np.datetime64(x[3], "D") for x in WAREHOUSES],
            "is_active": [x[4] for x in WAREHOUSES],
        }
    )


def _warehouse_for_region(w: World) -> dict[int, list[int]]:
    wh = w.tables["warehouses"]
    city_region = w.tables["cities"].set_index("city_id")["region_id"]
    by_region: dict[int, list[int]] = {}
    for r in wh.itertuples(index=False):
        by_region.setdefault(int(city_region[r.city_id]), []).append(int(r.warehouse_id))
    return by_region


def build_shipments(w: World) -> None:
    lp = w.params["logistics"]
    o = w.tables["orders"]
    items = w.tables["order_items"]
    rng = w.rng("shipments", "attrs")
    rng_late = w.rng("shipments", "late")
    st = o["status"].to_numpy()
    shipped = np.isin(st, SHIPPED_STATUSES)
    # one shipment per (order, seller)
    grp = items[["order_id", "seller_id"]].drop_duplicates()
    grp = grp[grp["order_id"].isin(o["order_id"].to_numpy()[shipped])].reset_index(drop=True)
    n = len(grp)
    if n != w.n("shipments"):
        raise RuntimeError(f"{n} shipments, expected {w.n('shipments')}")
    oi = grp["order_id"].to_numpy() - 1
    seq_in_order = grp.groupby("order_id").cumcount().to_numpy()
    addr = w.tables["customer_addresses"].set_index("address_id")
    cities = w.tables["cities"].set_index("city_id")
    dest_city = addr.loc[o["address_id"].to_numpy()[oi], "city_id"].to_numpy()
    dest_region = cities.loc[dest_city, "region_id"].to_numpy()
    zone = cities.loc[dest_city, "courier_zone"].to_numpy()
    order_utc = o["order_at"].to_numpy()[oi] - w.utc_offset
    end_utc = (w.end + DAY).astype("datetime64[s]") - w.utc_offset - np.timedelta64(1, "s")

    couriers = w.tables["couriers"]
    cnames = couriers["name"].to_numpy()
    share = np.array(lp["courier_share"], dtype=float)
    courier = rng.choice(len(cnames), size=n, p=share / share.sum())
    fastbox = int(np.flatnonzero(cnames == FASTBOX)[0])
    oday = o["order_at"].to_numpy().astype("datetime64[D]")[oi]
    chattogram = dest_region == 2
    march26 = (oday >= np.datetime64("2026-03-01")) & (oday <= np.datetime64("2026-03-31"))
    batch = chattogram & march26 & (rng.random(n) < 0.6)  # F02: the FastBox batch
    courier[batch] = fastbox

    wh = w.tables["warehouses"]
    by_region = _warehouse_for_region(w)
    wh_open = wh["opened_at"].to_numpy().astype("datetime64[D]")
    warehouse = np.empty(n, dtype=np.int64)
    closed = np.datetime64(BARISHAL_CLOSED, "D")
    for i in range(n):
        opts = [
            x
            for x in by_region.get(int(dest_region[i]), [])
            if wh_open[x - 1] <= oday[i] and not (WAREHOUSES[x - 1][0] == "Barishal-1" and oday[i] > closed)
        ]
        if not opts or rng.random() < 0.25:
            opts = [1, 2, 3, 4]
        warehouse[i] = opts[int(rng.integers(0, len(opts)))]

    ship_utc = order_utc + (rng.integers(4 * 3600, 40 * 3600, n) + seq_in_order * 86400).astype(
        "timedelta64[s]"
    )
    ship_utc = np.minimum(ship_utc, end_utc - np.timedelta64(3600, "s"))
    ship_utc = np.maximum(ship_utc, order_utc + np.timedelta64(600, "s"))
    ship_local_day = (ship_utc + w.utc_offset).astype("datetime64[D]")
    sla = np.array([lp["sla_days"][z] for z in zone])
    promised = ship_local_day + sla * DAY
    fee = np.array([lp["fee_bdt"][z] for z in zone], dtype=float)

    status_o = st[oi]
    delivered = np.isin(status_o, ["delivered", "returned"])
    late_rate = np.array([lp["courier_late_rate"][c] for c in cnames])[courier]
    # F01: Padma Gadgets' shipments delivered in 2026-Q3 run late much more often
    sellers = w.tables["sellers"]
    padma_id = int(sellers.loc[sellers["name"] == PADMA, "seller_id"].iloc[0])
    orders_with_padma = set(items.loc[items["seller_id"] == padma_id, "order_id"].tolist())
    has_padma = np.array([int(x) + 1 in orders_with_padma for x in oi])
    q3 = (promised >= np.datetime64("2026-07-01")) & (promised <= np.datetime64("2026-09-30"))
    late_rate = np.where(has_padma & q3, 0.42, late_rate)
    late = delivered & (rng_late.random(n) < late_rate)
    on_time_days = np.array([int(rng.integers(0, s + 1)) for s in sla])
    deliv_day = np.where(late, promised + rng.integers(1, 5, n) * DAY, ship_local_day + on_time_days * DAY)
    deliv_day = np.maximum(deliv_day, ship_local_day)
    tod = rng.integers(10 * 3600, 21 * 3600, n).astype("timedelta64[s]")
    deliv_utc = deliv_day.astype("datetime64[s]") + tod - w.utc_offset
    deliv_utc = np.maximum(deliv_utc, ship_utc + np.timedelta64(3 * 3600, "s"))
    # keep business-day lateness consistent after the clamp
    beyond = deliv_utc > end_utc
    if beyond.any():
        deliv_utc[beyond] = end_utc
    attempts = np.where(delivered, rng.choice([1, 2, 3], size=n, p=[0.85, 0.11, 0.04]), 1)
    df = pd.DataFrame(
        {
            "shipment_id": np.arange(1, n + 1),
            "order_id": oi + 1,
            "courier_id": courier + 1,
            "warehouse_id": warehouse,
            "shipped_at_utc": ship_utc,
            "promised_date": promised,
            "delivered_at_utc": np.where(delivered, deliv_utc, np.datetime64("NaT", "s")),
            "status": np.where(delivered, "delivered", "in_transit"),
            "shipping_fee_bdt": fee,
            "attempt_count": attempts,
        }
    )
    w.tables["shipments"] = df
    first_ship = np.full(len(o), np.datetime64("NaT", "s"), dtype="datetime64[s]")
    last_deliv = np.full(len(o), np.datetime64("NaT", "s"), dtype="datetime64[s]")
    s_df = pd.DataFrame(
        {"o": oi, "s": ship_utc, "d": np.where(delivered, deliv_utc, np.datetime64("NaT", "s"))}
    )
    g = s_df.groupby("o").agg(s=("s", "min"), d=("d", "max"))
    first_ship[g.index.to_numpy()] = g["s"].to_numpy()
    last_deliv[g.index.to_numpy()] = g["d"].to_numpy()
    w.meta["first_ship_utc"] = first_ship
    w.meta["last_delivered_utc"] = last_deliv
    w.meta["shipment_dest_city"] = dest_city
    w.meta["shipment_dest_region"] = dest_region


def build_shipment_events(w: World) -> None:
    s = w.tables["shipments"]
    n_target = w.n("shipment_events")
    rng = w.rng("shipment_events", "attrs")
    wh_city = w.tables["warehouses"].set_index("warehouse_id")["city_id"]
    dest = w.meta["shipment_dest_city"]
    ship = s["shipped_at_utc"].to_numpy()
    deliv = s["delivered_at_utc"].to_numpy()
    attempts = s["attempt_count"].to_numpy()
    is_deliv = s["status"].to_numpy() == "delivered"
    wcity = wh_city.loc[s["warehouse_id"].to_numpy()].to_numpy()
    n = len(s)
    base = n + int(is_deliv.sum()) + int((attempts - 1)[is_deliv].sum())
    optional_cap = n + int(is_deliv.sum())
    extra = n_target - base
    if not 0 <= extra <= optional_cap:
        raise RuntimeError(f"cannot reach {n_target} shipment events (base {base})")
    # optional slots: 0..n-1 = arrived_hub for each shipment; n.. = out_for_delivery for delivered ones
    deliv_idx = np.flatnonzero(is_deliv)
    chosen = rng.choice(optional_cap, size=extra, replace=False)
    hub = np.zeros(n, dtype=bool)
    ofd = np.zeros(n, dtype=bool)
    hub[chosen[chosen < n]] = True
    ofd[deliv_idx[chosen[chosen >= n] - n]] = True
    rows_s, rows_t, rows_e, rows_c, rows_n = [], [], [], [], []

    def note(kind: str) -> object:
        opts = SHIPMENT_NOTES[kind]
        return opts[int(rng.integers(0, len(opts)))]

    for i in range(n):
        t0 = ship[i]
        ev = [("picked_up", t0, wcity[i])]
        if is_deliv[i]:
            t_end = deliv[i]
            span = max(int((t_end - t0) / np.timedelta64(1, "s")), 60)
            mids = np.sort(rng.integers(1, span, size=int(hub[i]) + int(ofd[i]) + int(attempts[i] - 1)))
            k = 0
            if hub[i]:
                ev.append(("arrived_hub", t0 + np.timedelta64(int(mids[k]), "s"), wcity[i]))
                k += 1
            for _ in range(int(attempts[i] - 1)):
                ev.append(("failed_attempt", t0 + np.timedelta64(int(mids[k]), "s"), dest[i]))
                k += 1
            if ofd[i]:
                ev.append(("out_for_delivery", t0 + np.timedelta64(int(mids[k]), "s"), dest[i]))
            ev.append(("delivered", t_end, dest[i]))
            ev.sort(key=lambda x: x[1])
        elif hub[i]:
            ev.append(("arrived_hub", t0 + np.timedelta64(int(rng.integers(1800, 36000)), "s"), wcity[i]))
        for e, t, c in ev:
            rows_s.append(i + 1)
            rows_e.append(e)
            rows_t.append(t)
            rows_c.append(int(c))
            rows_n.append(note(e))
    end_utc = (w.end + DAY).astype("datetime64[s]") - w.utc_offset - np.timedelta64(1, "s")
    t_arr = np.minimum(np.array(rows_t, dtype="datetime64[s]"), end_utc)
    df = pd.DataFrame(
        {
            "shipment_id": rows_s,
            "event_type": rows_e,
            "event_at_utc": t_arr,
            "city_id": rows_c,
            "note": rows_n,
        }
    )
    df.insert(0, "shipment_event_id", np.arange(1, len(df) + 1))
    w.tables["shipment_events"] = df
