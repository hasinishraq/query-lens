"""Rate limiting for free tiers: RPM, TPM, RPD, TPD and monthly tokens per provider/model.

Every request is written to a persistent SQLite ledger (``data/quota.sqlite``), so limits survive restarts
and a run can span several days. Minute limits are handled by **waiting**; when a daily or monthly limit
is reached, :class:`QuotaExhausted` is raised so the runner can stop and ``--resume`` later.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import Limits, Provider

log = logging.getLogger(__name__)


class QuotaExhausted(RuntimeError):
    """A daily or monthly free-tier limit is used up for this provider/model."""

    def __init__(self, provider: str, model: str, which: str, resets: str) -> None:
        super().__init__(f"{provider}/{model}: {which} limit reached; resets {resets}")
        self.provider, self.model, self.which, self.resets = provider, model, which, resets


_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL, model TEXT NOT NULL, ts REAL NOT NULL,
  day TEXT NOT NULL, month TEXT NOT NULL,
  requests INTEGER NOT NULL DEFAULT 1, tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS usage_pm_ts ON usage (provider, model, ts);
CREATE INDEX IF NOT EXISTS usage_p_day ON usage (provider, day);
"""


@dataclass
class Usage:
    provider: str
    model: str
    requests_today: int
    tokens_today: int
    tokens_month: int
    limits: Limits


class RateLimiter:
    def __init__(
        self,
        db_path: str | Path,
        providers: dict[str, Provider],
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        max_wait_s: float = 600.0,
    ) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.providers = providers
        self.clock = clock
        self.sleep = sleep
        self.max_wait_s = max_wait_s
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def _day_month(self, provider: str, ts: float) -> tuple[str, str]:
        tz = ZoneInfo(self.providers[provider].daily_reset_tz)
        dt = datetime.fromtimestamp(ts, tz)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")

    def limits_for(self, provider: str, model: str) -> Limits:
        p = self.providers[provider]
        try:
            return p.model(model).limits
        except KeyError:
            return Limits()

    def _window(
        self, c: sqlite3.Connection, provider: str, model: str | None, since: float
    ) -> tuple[int, int]:
        if model is None:
            row = c.execute(
                "SELECT COALESCE(SUM(requests),0), COALESCE(SUM(tokens),0) FROM usage WHERE provider=? AND ts>=?",
                (provider, since),
            ).fetchone()
        else:
            row = c.execute(
                "SELECT COALESCE(SUM(requests),0), COALESCE(SUM(tokens),0) FROM usage "
                "WHERE provider=? AND model=? AND ts>=?",
                (provider, model, since),
            ).fetchone()
        return int(row[0]), int(row[1])

    def _period(
        self, c: sqlite3.Connection, provider: str, model: str | None, col: str, val: str
    ) -> tuple[int, int]:
        q = f"SELECT COALESCE(SUM(requests),0), COALESCE(SUM(tokens),0) FROM usage WHERE provider=? AND {col}=?"
        args: tuple[str, ...] = (provider, val)
        if model is not None:
            q += " AND model=?"
            args = (provider, val, model)
        row = c.execute(q, args).fetchone()
        return int(row[0]), int(row[1])

    def _check(self, provider: str, model: str, est_tokens: int) -> float:
        """Seconds to wait before the request fits (0 = go). Raises QuotaExhausted for daily/monthly limits."""
        now = self.clock()
        day, month = self._day_month(provider, now)
        lim = self.limits_for(provider, model)
        acct = self.providers[provider].account_limits
        with self._conn() as c:
            for scope_model, limits in ((model, lim), (None, acct)):
                if limits is None:
                    continue
                req_d, tok_d = self._period(c, provider, scope_model, "day", day)
                if limits.rpd is not None and req_d + 1 > limits.rpd:
                    raise QuotaExhausted(
                        provider,
                        model,
                        "requests/day",
                        f"after {day} ({self.providers[provider].daily_reset_tz})",
                    )
                if limits.tpd is not None and tok_d + est_tokens > limits.tpd:
                    raise QuotaExhausted(provider, model, "tokens/day", f"after {day}")
                if limits.tokens_per_month is not None:
                    _, tok_m = self._period(c, provider, scope_model, "month", month)
                    if tok_m + est_tokens > limits.tokens_per_month:
                        raise QuotaExhausted(provider, model, "tokens/month", f"after {month}")
                req_m, tok_min = self._window(c, provider, scope_model, now - 60)
                over_rpm = limits.rpm is not None and req_m + 1 > limits.rpm
                over_tpm = limits.tpm is not None and tok_min + est_tokens > limits.tpm and tok_min > 0
                if over_rpm or over_tpm:
                    oldest = c.execute(
                        "SELECT MIN(ts) FROM usage WHERE provider=? AND ts>=?"
                        + (" AND model=?" if scope_model else ""),
                        (provider, now - 60, scope_model) if scope_model else (provider, now - 60),
                    ).fetchone()[0]
                    return max(0.5, (oldest or now) + 60 - now)
        return 0.0

    def acquire(self, provider: str, model: str, est_tokens: int = 0) -> int:
        """Block until the request fits the limits, then reserve it in the ledger. Returns the ledger row id."""
        waited = 0.0
        while True:
            with self._lock:
                wait = self._check(provider, model, est_tokens)
                if wait <= 0:
                    now = self.clock()
                    day, month = self._day_month(provider, now)
                    with self._conn() as c:
                        cur = c.execute(
                            "INSERT INTO usage (provider, model, ts, day, month, requests, tokens) "
                            "VALUES (?,?,?,?,?,1,?)",
                            (provider, model, now, day, month, est_tokens),
                        )
                        return int(cur.lastrowid or 0)
            if waited + wait > self.max_wait_s:
                raise QuotaExhausted(provider, model, "minute window (waited too long)", "in a minute")
            log.info("rate limit: waiting %.1fs for %s/%s", wait, provider, model)
            self.sleep(wait)
            waited += wait

    def record(self, row_id: int, tokens: int) -> None:
        """Replace the reserved token estimate with the actual count."""
        with self._conn() as c:
            c.execute("UPDATE usage SET tokens=? WHERE id=?", (int(tokens), row_id))

    def usage_today(self) -> list[Usage]:
        out: list[Usage] = []
        now = self.clock()
        with self._conn() as c:
            for pname, p in self.providers.items():
                day, month = self._day_month(pname, now)
                for m in p.models:
                    req, tok = self._period(c, pname, m.id, "day", day)
                    _, tok_m = self._period(c, pname, m.id, "month", month)
                    out.append(Usage(pname, m.id, req, tok, tok_m, m.limits))
        return out
