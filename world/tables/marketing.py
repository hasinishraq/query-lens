"""Marketing: campaigns and promotions (fixed in both sizes); redemptions are built with orders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import DAY, World
from ..rng import choice

CHANNELS = ["facebook", "google", "youtube", "tiktok", "email", "sms", "influencer", "tv"]
OBJECTIVES = ["acquisition", "retention", "awareness", "sales"]
TEAMS = ["growth", "crm", "brand", "category"]

# (name stem, month, day, duration days)
THEMED = [
    ("NewYear", 1, 1, 10),
    ("Valentine", 2, 7, 8),
    ("Ekushey", 2, 15, 8),
    ("Ramadan", 0, 0, 0),
    ("Boishakh", 4, 1, 20),
    ("EidFitr", 0, 0, 0),
    ("EidAdha", 0, 0, 0),
    ("Monsoon", 7, 1, 21),
    ("BackToSchool", 1, 10, 14),
    ("Puja", 10, 1, 14),
    ("Mega1111_", 11, 1, 12),
    ("Winter", 12, 1, 25),
    ("Victory", 12, 10, 8),
    ("YearEnd", 12, 20, 12),
]


def build_campaigns(w: World) -> None:
    rng = w.rng("marketing_campaigns", "attrs")
    rows: list[tuple] = []
    start, end = w.start, w.end
    eid_f = [np.datetime64(str(d), "D") for d in w.params["holidays"]["eid_al_fitr"]]
    eid_a = [np.datetime64(str(d), "D") for d in w.params["holidays"]["eid_al_adha"]]
    for yi, year in enumerate((2024, 2025, 2026)):
        yy = str(year)[2:]
        for stem, m, d, dur in THEMED:
            if stem == "Ramadan":
                s = eid_f[yi] - 30 * DAY
                e = eid_f[yi] - 8 * DAY
            elif stem == "EidFitr":
                s = eid_f[yi] - 10 * DAY
                e = eid_f[yi] - DAY
            elif stem == "EidAdha":
                s = eid_a[yi] - 10 * DAY
                e = eid_a[yi] - DAY
            else:
                s = np.datetime64(f"{year}-{m:02d}-{d:02d}", "D")
                e = s + (dur - 1) * DAY
            if s < start or e > end:
                continue
            rows.append((f"{stem}{yy}", s, e))
    themed_n = len(rows)
    # always-on campaigns fill the remainder (fixed total 120)
    months = np.arange(start.astype("datetime64[M]"), (end + DAY).astype("datetime64[M]"))
    n_fill = 120 - themed_n
    chans = ["google", "facebook", "tiktok", "email"]
    k = 0
    while len(rows) < 120:
        mo = months[k % len(months)]
        ch = chans[(k // len(months)) % len(chans)]
        s = mo.astype("datetime64[D]")
        e = (mo + 1).astype("datetime64[D]") - DAY
        rows.append((f"AlwaysOn-{ch}-{str(mo).replace('-', '')}", s, e))
        k += 1
    assert len(rows) == 120 and n_fill >= 0
    n = len(rows)
    chan = choice(rng, CHANNELS, [0.3, 0.22, 0.1, 0.1, 0.08, 0.08, 0.07, 0.05], n)
    for i, (name, _, _) in enumerate(rows):
        if name.startswith("AlwaysOn-"):
            chan[i] = name.split("-")[1]
    obj = choice(rng, OBJECTIVES, [0.35, 0.2, 0.15, 0.3], n)
    df = pd.DataFrame(
        {
            "campaign_id": np.arange(1, n + 1),
            "name": [r[0] for r in rows],
            "channel": chan,
            "start_date": [r[1] for r in rows],
            "end_date": [r[2] for r in rows],
            "budget_bdt": np.round(rng.lognormal(13.2, 0.7, n), -3),
            "objective": obj,
            "owner_team": choice(rng, TEAMS, [0.4, 0.2, 0.2, 0.2], n),
        }
    )
    # acquisition campaigns: Boishakh25 is an acquisition campaign (F06)
    df.loc[df["name"] == "Boishakh25", "objective"] = "acquisition"
    w.tables["marketing_campaigns"] = df


def build_promotions(w: World) -> None:
    rng = w.rng("promotions", "attrs")
    camp = w.tables["marketing_campaigns"]
    n = 400
    counts = np.full(len(camp), n // len(camp))
    counts[: n - counts.sum()] += 1
    rows = []
    pid = 1
    for (_, c), k in zip(camp.iterrows(), counts, strict=True):
        s = np.datetime64(c["start_date"], "D")
        e = np.datetime64(c["end_date"], "D")
        span = int((e - s) / DAY) + 1
        stem = (
            "".join(ch for ch in str(c["name"]).upper() if ch.isalnum())[:6] + f"{int(c['campaign_id']):03d}"
        )
        for j in range(int(k)):
            ptype = "percent" if rng.random() < 0.6 else "flat"
            val = (
                float(rng.choice([5, 10, 12, 15, 20, 25]))
                if ptype == "percent"
                else float(rng.choice([50, 100, 150, 200, 300, 500]))
            )
            min_order = float(rng.choice([0, 300, 500, 1000, 1500, 2000]))
            off = int(rng.integers(0, max(1, span // 2)))
            ps = s + off * DAY
            pe = min(e, ps + int(rng.integers(3, max(4, span))) * DAY)
            rows.append(
                (
                    pid,
                    int(c["campaign_id"]),
                    f"{stem}{j + 1:02d}",
                    ptype,
                    val,
                    min_order,
                    ps.astype("datetime64[s]"),
                    (pe + DAY).astype("datetime64[s]") - np.timedelta64(1, "s"),
                    int(rng.choice([500, 1000, 2000, 5000])),
                )
            )
            pid += 1
    w.tables["promotions"] = pd.DataFrame(
        rows,
        columns=[
            "promo_id",
            "campaign_id",
            "code",
            "promo_type",
            "discount_value",
            "min_order_bdt",
            "start_at",
            "end_at",
            "max_redemptions",
        ],
    )


def build(w: World) -> None:
    build_campaigns(w)
    build_promotions(w)
