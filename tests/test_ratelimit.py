import pytest

from querylens.config import Limits, ModelSpec, Provider
from querylens.ratelimit import QuotaExhausted, RateLimiter


class Clock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _prov(**limits):
    return {
        "p": Provider(
            name="p",
            base_url="http://x",
            key_env=None,
            family="f",
            daily_reset_tz="America/Los_Angeles",
            models=[ModelSpec(id="m", limits=Limits(**limits))],
        )
    }


def test_rpm_waits_instead_of_failing(tmp_path):
    clock = Clock()
    lim = RateLimiter(tmp_path / "q.sqlite", _prov(rpm=2), clock=clock, sleep=clock.sleep)
    t0 = clock.t
    for _ in range(3):
        lim.acquire("p", "m")
    assert clock.t - t0 >= 59  # third request waited for the minute window


def test_tpm_waits(tmp_path):
    clock = Clock()
    lim = RateLimiter(tmp_path / "q.sqlite", _prov(tpm=1000), clock=clock, sleep=clock.sleep)
    row = lim.acquire("p", "m", 800)
    lim.record(row, 900)
    t0 = clock.t
    lim.acquire("p", "m", 500)
    assert clock.t > t0


def test_rpd_raises_and_persists(tmp_path):
    clock = Clock()
    db = tmp_path / "q.sqlite"
    lim = RateLimiter(db, _prov(rpd=2), clock=clock, sleep=clock.sleep)
    lim.acquire("p", "m")
    lim.acquire("p", "m")
    lim2 = RateLimiter(db, _prov(rpd=2), clock=clock, sleep=clock.sleep)  # restart: ledger survives
    with pytest.raises(QuotaExhausted) as e:
        lim2.acquire("p", "m")
    assert e.value.which == "requests/day"
    clock.t += 86_400  # next day (Pacific)
    lim2.acquire("p", "m")


def test_tpd_and_monthly(tmp_path):
    clock = Clock()
    lim = RateLimiter(
        tmp_path / "q.sqlite", _prov(tpd=1000, tokens_per_month=1500), clock=clock, sleep=clock.sleep
    )
    lim.record(lim.acquire("p", "m", 10), 900)
    with pytest.raises(QuotaExhausted):
        lim.acquire("p", "m", 200)
    clock.t += 86_400
    lim.record(lim.acquire("p", "m", 10), 500)
    with pytest.raises(QuotaExhausted) as e:
        lim.acquire("p", "m", 200)
    assert e.value.which in ("tokens/month", "tokens/day")


def test_account_limits(tmp_path):
    clock = Clock()
    provs = _prov()
    provs["p"].account_limits = Limits(rpd=1)
    lim = RateLimiter(tmp_path / "q.sqlite", provs, clock=clock, sleep=clock.sleep)
    lim.acquire("p", "m")
    with pytest.raises(QuotaExhausted):
        lim.acquire("p", "m")


def test_usage_report(tmp_path):
    clock = Clock()
    lim = RateLimiter(tmp_path / "q.sqlite", _prov(rpd=10), clock=clock, sleep=clock.sleep)
    lim.record(lim.acquire("p", "m", 5), 42)
    u = lim.usage_today()[0]
    assert (u.requests_today, u.tokens_today) == (1, 42)
