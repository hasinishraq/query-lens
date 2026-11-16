"""Reference tables: regions, cities, calendar_dim, currency_rates (fixed in both sizes)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import DAY, World

# division -> (Bangla name, population in millions, [(district, Bangla name)])
DIVISIONS: list[tuple[str, str, float, list[tuple[str, str]]]] = [
    (
        "Dhaka",
        "ঢাকা",
        44.2,
        [
            ("Dhaka", "ঢাকা"),
            ("Gazipur", "গাজীপুর"),
            ("Narayanganj", "নারায়ণগঞ্জ"),
            ("Narsingdi", "নরসিংদী"),
            ("Munshiganj", "মুন্সীগঞ্জ"),
            ("Manikganj", "মানিকগঞ্জ"),
            ("Tangail", "টাঙ্গাইল"),
            ("Kishoreganj", "কিশোরগঞ্জ"),
            ("Faridpur", "ফরিদপুর"),
            ("Gopalganj", "গোপালগঞ্জ"),
            ("Madaripur", "মাদারীপুর"),
            ("Rajbari", "রাজবাড়ী"),
            ("Shariatpur", "শরীয়তপুর"),
        ],
    ),
    (
        "Chattogram",
        "চট্টগ্রাম",
        33.2,
        [
            ("Chattogram", "চট্টগ্রাম"),
            ("Cox's Bazar", "কক্সবাজার"),
            ("Cumilla", "কুমিল্লা"),
            ("Feni", "ফেনী"),
            ("Noakhali", "নোয়াখালী"),
            ("Lakshmipur", "লক্ষ্মীপুর"),
            ("Chandpur", "চাঁদপুর"),
            ("Brahmanbaria", "ব্রাহ্মণবাড়িয়া"),
            ("Rangamati", "রাঙ্গামাটি"),
            ("Khagrachhari", "খাগড়াছড়ি"),
            ("Bandarban", "বান্দরবান"),
        ],
    ),
    (
        "Rajshahi",
        "রাজশাহী",
        20.4,
        [
            ("Rajshahi", "রাজশাহী"),
            ("Bogura", "বগুড়া"),
            ("Pabna", "পাবনা"),
            ("Sirajganj", "সিরাজগঞ্জ"),
            ("Natore", "নাটোর"),
            ("Naogaon", "নওগাঁ"),
            ("Chapainawabganj", "চাঁপাইনবাবগঞ্জ"),
            ("Joypurhat", "জয়পুরহাট"),
        ],
    ),
    (
        "Khulna",
        "খুলনা",
        17.4,
        [
            ("Khulna", "খুলনা"),
            ("Jashore", "যশোর"),
            ("Satkhira", "সাতক্ষীরা"),
            ("Bagerhat", "বাগেরহাট"),
            ("Kushtia", "কুষ্টিয়া"),
            ("Chuadanga", "চুয়াডাঙ্গা"),
            ("Meherpur", "মেহেরপুর"),
            ("Jhenaidah", "ঝিনাইদহ"),
            ("Magura", "মাগুরা"),
            ("Narail", "নড়াইল"),
        ],
    ),
    (
        "Barishal",
        "বরিশাল",
        9.1,
        [
            ("Barishal", "বরিশাল"),
            ("Patuakhali", "পটুয়াখালী"),
            ("Bhola", "ভোলা"),
            ("Pirojpur", "পিরোজপুর"),
            ("Barguna", "বরগুনা"),
            ("Jhalokathi", "ঝালকাঠি"),
        ],
    ),
    (
        "Sylhet",
        "সিলেট",
        11.0,
        [
            ("Sylhet", "সিলেট"),
            ("Moulvibazar", "মৌলভীবাজার"),
            ("Habiganj", "হবিগঞ্জ"),
            ("Sunamganj", "সুনামগঞ্জ"),
        ],
    ),
    (
        "Rangpur",
        "রংপুর",
        17.6,
        [
            ("Rangpur", "রংপুর"),
            ("Dinajpur", "দিনাজপুর"),
            ("Kurigram", "কুড়িগ্রাম"),
            ("Gaibandha", "গাইবান্ধা"),
            ("Nilphamari", "নীলফামারী"),
            ("Lalmonirhat", "লালমনিরহাট"),
            ("Thakurgaon", "ঠাকুরগাঁও"),
            ("Panchagarh", "পঞ্চগড়"),
        ],
    ),
    (
        "Mymensingh",
        "ময়মনসিংহ",
        12.2,
        [
            ("Mymensingh", "ময়মনসিংহ"),
            ("Jamalpur", "জামালপুর"),
            ("Netrokona", "নেত্রকোণা"),
            ("Sherpur", "শেরপুর"),
        ],
    ),
]

METROS = {
    "Dhaka",
    "Chattogram",
    "Khulna",
    "Rajshahi",
    "Sylhet",
    "Barishal",
    "Rangpur",
    "Mymensingh",
    "Gazipur",
    "Narayanganj",
    "Cumilla",
}
DHAKA_SUBURBS = {"Gazipur", "Narayanganj", "Narsingdi", "Munshiganj", "Manikganj"}

# relative e-commerce weight of each district (customers and sellers concentrate in metros)
CITY_WEIGHT_METRO = {
    "Dhaka": 30.0,
    "Chattogram": 9.0,
    "Gazipur": 5.0,
    "Narayanganj": 4.0,
    "Cumilla": 2.5,
    "Khulna": 2.5,
    "Rajshahi": 2.2,
    "Sylhet": 2.4,
    "Barishal": 1.4,
    "Rangpur": 1.6,
    "Mymensingh": 1.6,
}

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_regions(w: World) -> None:
    rows = [(i + 1, d[0], d[1], d[2]) for i, d in enumerate(DIVISIONS)]
    w.tables["regions"] = pd.DataFrame(rows, columns=["region_id", "name", "name_bn", "population_m"])


def build_cities(w: World) -> None:
    rows = []
    weights = []
    cid = 1
    for rid, (_, _, _, districts) in enumerate(DIVISIONS, start=1):
        for name, name_bn in districts:
            if name == "Dhaka":
                zone = "inside_dhaka"
            elif name in DHAKA_SUBURBS:
                zone = "dhaka_suburb"
            else:
                zone = "outside_dhaka"
            rows.append((cid, rid, name, name_bn, name in METROS, zone))
            weights.append(CITY_WEIGHT_METRO.get(name, 0.55))
            cid += 1
    df = pd.DataFrame(rows, columns=["city_id", "region_id", "name", "name_bn", "is_metro", "courier_zone"])
    w.tables["cities"] = df
    wt = np.asarray(weights)
    w.meta["city_weight"] = wt / wt.sum()


def _holidays(w: World) -> dict[np.datetime64, str]:
    hol: dict[np.datetime64, str] = {}
    days = w.days
    years = sorted({int(str(d)[:4]) for d in days})
    for y in years:
        for md, name in w.params["holidays"]["fixed"].items():
            hol[np.datetime64(f"{y}-{md}", "D")] = name
        hol[np.datetime64(f"{y}-11-11", "D")] = "11.11 Mega Sale"
    for d in w.params["holidays"]["eid_al_fitr"]:
        for k in range(3):
            hol[np.datetime64(str(d), "D") + k * DAY] = "Eid al-Fitr"
    for d in w.params["holidays"]["eid_al_adha"]:
        for k in range(3):
            hol[np.datetime64(str(d), "D") + k * DAY] = "Eid al-Adha"
    return {d: n for d, n in hol.items() if w.start <= d <= w.end}


def build_calendar(w: World) -> None:
    days = w.days
    hol = _holidays(w)
    ts = pd.to_datetime(days)
    dow = ts.dayofweek.to_numpy()
    month = ts.month.to_numpy()
    year = ts.year.to_numpy()
    df = pd.DataFrame(
        {
            "date_key": (year * 10000 + month * 100 + ts.day.to_numpy()).astype(np.int64),
            "date": days,
            "year": year,
            "quarter": (month - 1) // 3 + 1,
            "month": month,
            "month_name": [MONTH_NAMES[m - 1] for m in month],
            "week_of_year": ts.isocalendar().week.to_numpy().astype(np.int64),
            "day_of_week": [DOW[d] for d in dow],
            "is_weekend": np.isin(dow, [4, 5]),  # Friday, Saturday
            "is_holiday": [d in hol for d in days],
            "holiday_name": [hol.get(d) for d in days],
            "fiscal_year": np.where(
                month >= 7, year + 1, year
            ),  # FY ends in June; FY2026 = Jul 2025-Jun 2026
        }
    )
    w.tables["calendar_dim"] = df
    w.meta["holidays"] = hol
    _day_weights(w, dow)


def _day_weights(w: World, dow: np.ndarray) -> None:
    """Relative order volume per day: growth x weekly cycle x seasonality."""
    p = w.params["orders"]
    days = w.days
    months = days.astype("datetime64[M]").astype(int) - w.start.astype("datetime64[M]").astype(int)
    frac_month = np.arange(len(days)) / 30.4375
    wt = (1 + p["monthly_growth"]) ** frac_month
    wt = wt * np.where(np.isin(dow, [4, 5]), p["weekend_multiplier"], 1.0)
    s = p["seasonality"]
    for d in w.params["holidays"]["eid_al_fitr"]:
        e = np.datetime64(str(d), "D")
        wt[(days >= e - 7 * DAY) & (days < e)] *= s["eid_al_fitr_week"]
    for d in w.params["holidays"]["eid_al_adha"]:
        e = np.datetime64(str(d), "D")
        wt[(days >= e - 7 * DAY) & (days < e)] *= s["eid_al_adha_week"]
    md = np.array([str(d)[5:] for d in days])
    wt[md == "11-11"] *= s["sale_1111"]
    wt[md == "04-14"] *= s["pohela_boishakh"]
    w.meta["day_weight"] = wt
    w.meta["month_of_day"] = months


def build_currency_rates(w: World) -> None:
    rng = w.rng("currency_rates", "walk")
    days = w.days
    steps = rng.normal(0.012, 0.18, size=len(days))
    rate = 109.5 + np.cumsum(steps)
    w.tables["currency_rates"] = pd.DataFrame(
        {
            "rate_date": days,
            "currency": "USD",
            "bdt_per_unit": np.round(rate, 4),
            "source": "bangladesh_bank",
        }
    )


def build(w: World) -> None:
    build_regions(w)
    build_cities(w)
    build_calendar(w)
    build_currency_rates(w)
