"""Shared helpers for eval-item templates: periods, Bangla/Banglish phrasing, SQL fragments."""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
MONTHS_EN = list(calendar.month_name)[1:]
MONTHS_BN = [
    "জানুয়ারি",
    "ফেব্রুয়ারি",
    "মার্চ",
    "এপ্রিল",
    "মে",
    "জুন",
    "জুলাই",
    "আগস্ট",
    "সেপ্টেম্বর",
    "অক্টোবর",
    "নভেম্বর",
    "ডিসেম্বর",
]
Q_BN = {1: "প্রথম", 2: "দ্বিতীয়", 3: "তৃতীয়", 4: "চতুর্থ"}


def bn(x: Any) -> str:
    return str(x).translate(BN_DIGITS)


@dataclass(frozen=True)
class Period:
    key: str
    en: str
    bn: str
    bl: str  # Banglish
    start: date
    end: date

    @property
    def s(self) -> str:
        return self.start.isoformat()

    @property
    def e(self) -> str:
        return self.end.isoformat()

    def between(self, expr: str) -> str:
        return f"{expr} BETWEEN DATE '{self.s}' AND DATE '{self.e}'"


def month(y: int, m: int) -> Period:
    last = calendar.monthrange(y, m)[1]
    return Period(
        f"{y}-{m:02d}",
        f"{MONTHS_EN[m - 1]} {y}",
        f"{bn(y)} সালের {MONTHS_BN[m - 1]} মাসে",
        f"{y} er {MONTHS_EN[m - 1]} mashe",
        date(y, m, 1),
        date(y, m, last),
    )


def quarter(y: int, q: int) -> Period:
    sm = 3 * (q - 1) + 1
    end = date(y, sm + 2, calendar.monthrange(y, sm + 2)[1])
    return Period(
        f"{y}Q{q}", f"{y} Q{q}", f"{bn(y)} সালের {Q_BN[q]} প্রান্তিকে", f"{y} er Q{q} te", date(y, sm, 1), end
    )


def year(y: int) -> Period:
    end = date(2026, 10, 31) if y == 2026 else date(y, 12, 31)
    en = f"{y}" if y < 2026 else "2026 (January to October)"
    return Period(f"{y}", en, f"{bn(y)} সালে", f"{y} e", date(y, 1, 1), end)


def fiscal(fy: int) -> Period:
    return Period(
        f"FY{fy}",
        f"fiscal year {fy - 1}-{str(fy)[2:]} (July {fy - 1} to June {fy})",
        f"{bn(fy - 1)}-{bn(str(fy)[2:])} অর্থবছরে (জুলাই থেকে জুন)",
        f"FY {fy - 1}-{str(fy)[2:]} e",
        date(fy - 1, 7, 1),
        date(fy, 6, 30),
    )


def day(d: date) -> Period:
    return Period(
        d.isoformat(),
        f"{d.day} {MONTHS_EN[d.month - 1]} {d.year}",
        f"{bn(d.year)} সালের {bn(d.day)} {MONTHS_BN[d.month - 1]}",
        f"{d.day} {MONTHS_EN[d.month - 1]} {d.year} e",
        d,
        d,
    )


def span(key: str, en: str, bn_: str, bl: str, s: date, e: date) -> Period:
    return Period(key, en, bn_, bl, s, e)


MONTHS = [month(y, m) for y in (2024, 2025, 2026) for m in range(1, 13) if not (y == 2026 and m > 10)]
QUARTERS = [quarter(y, q) for y in (2024, 2025, 2026) for q in (1, 2, 3, 4) if not (y == 2026 and q == 4)]
YEARS = [year(2024), year(2025), year(2026)]
FISCAL = [fiscal(2025), fiscal(2026)]
ALL_PERIODS = MONTHS + QUARTERS + YEARS + FISCAL


def pick(rng: np.random.Generator, seq: list[Any], k: int) -> list[Any]:
    """k distinct items (or all, cycled with replacement beyond len)."""
    if k <= len(seq):
        idx = rng.choice(len(seq), size=k, replace=False)
        return [seq[i] for i in sorted(idx)]
    return [seq[i % len(seq)] for i in range(k)]


def days_in(p: Period) -> list[date]:
    out = []
    d = p.start
    while d <= p.end:
        out.append(d)
        d += timedelta(days=1)
    return out


@dataclass
class Instance:
    key: str  # unique per template (parameters)
    en: str
    bn: str
    bl: str
    gold: str
    alt: str
    trap: str | None = None
    ordered: bool = False
    must_mention: list[str] = field(default_factory=list)
    planted_fact: str | None = None
    difficulty: str | None = None  # override


# ---------------------------------------------------------------- SQL fragments
VALID = "NOT o.is_deleted AND o.status NOT IN ('cancelled', 'canceled', 'payment_failed')"
VALID_ALT = "o.is_deleted = FALSE AND o.status IN ('delivered', 'in_transit', 'returned')"
ORDER_VALUE = "COALESCE(o.total_bdt, o.total_amount * cr.bdt_per_unit)"
FX_JOIN = (
    "LEFT JOIN currency_rates cr ON cr.rate_date = CAST(o.order_at AS DATE) AND cr.currency = o.currency"
)
REGION_JOIN = (
    "JOIN customer_addresses a ON a.address_id = o.address_id JOIN cities ci ON ci.city_id = a.city_id "
    "JOIN regions r ON r.region_id = ci.region_id"
)
DHAKA = "INTERVAL 6 HOUR"


def odate(p: Period, alias: str = "o") -> str:
    return p.between(f"CAST({alias}.order_at AS DATE)")


def odate_alt(p: Period, alias: str = "o") -> str:
    nxt = p.end + timedelta(days=1)
    return f"{alias}.order_at >= TIMESTAMP '{p.s} 00:00:00' AND {alias}.order_at < TIMESTAMP '{nxt.isoformat()} 00:00:00'"


def utc_date(col: str) -> str:
    return f"CAST({col} + {DHAKA} AS DATE)"


# periods that include the last days of the data, where undelivered shipments (NULL delivered_at) exist
RECENT = [
    month(2026, 10),
    year(2026),
    span(
        "2026-09+10",
        "September and October 2026",
        "২০২৬ সালের সেপ্টেম্বর ও অক্টোবরে",
        "2026 er September ar October e",
        date(2026, 9, 1),
        date(2026, 10, 31),
    ),
    span(
        "FY2027",
        "fiscal year 2026-27 so far (July to October 2026)",
        "২০২৬-২৭ অর্থবছরে এখন পর্যন্ত (জুলাই-অক্টোবর ২০২৬)",
        "FY 2026-27 e ekhon porjonto",
        date(2026, 7, 1),
        date(2026, 10, 31),
    ),
    span(
        "last30",
        "the last 30 days of data (2 to 31 October 2026)",
        "ডেটার শেষ ৩০ দিনে (২-৩১ অক্টোবর ২০২৬)",
        "data r shesh 30 din e (2-31 October 2026)",
        date(2026, 10, 2),
        date(2026, 10, 31),
    ),
    span(
        "2026-08..10",
        "August to October 2026",
        "২০২৬ সালের আগস্ট থেকে অক্টোবর",
        "2026 er August theke October",
        date(2026, 8, 1),
        date(2026, 10, 31),
    ),
    span(
        "2026H2",
        "the second half of 2026 so far",
        "২০২৬ সালের দ্বিতীয়ার্ধে এখন পর্যন্ত",
        "2026 er second half e ekhon porjonto",
        date(2026, 7, 1),
        date(2026, 10, 31),
    ),
    span(
        "2026-10-15..31",
        "15 to 31 October 2026",
        "১৫-৩১ অক্টোবর ২০২৬",
        "15-31 October 2026",
        date(2026, 10, 15),
        date(2026, 10, 31),
    ),
]
DAYS = [day(date(2024 + (i % 3), 1 + (i * 5) % 12, 1 + (i * 7) % 28)) for i in range(40)]
DAYS = [d for d in DAYS if d.start <= date(2026, 10, 31)]
