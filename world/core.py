"""Shared state and time helpers for the BazaarDW generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .rng import rng_for
from .schema import target_rows

WORLD_DIR = Path(__file__).resolve().parent

DAY = np.timedelta64(1, "D")
HOUR = np.timedelta64(1, "h")
MINUTE = np.timedelta64(1, "m")
SECOND = np.timedelta64(1, "s")
FAR_FUTURE = np.datetime64("9999-12-31", "D")


def load_params() -> dict[str, Any]:
    with open(WORLD_DIR / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class World:
    seed: int
    size: str
    params: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def rng(self, table: str, purpose: str) -> np.random.Generator:
        return rng_for(self.seed, table, purpose)

    def n(self, table: str) -> int:
        return target_rows(table, self.size)

    @property
    def scale(self) -> float:
        return 0.1 if self.size == "small" else 1.0

    @property
    def start(self) -> np.datetime64:
        return np.datetime64(str(self.params["period"]["start"]), "D")

    @property
    def end(self) -> np.datetime64:
        return np.datetime64(str(self.params["period"]["end"]), "D")

    @property
    def days(self) -> np.ndarray:
        return np.arange(self.start, self.end + DAY, DAY)

    @property
    def utc_offset(self) -> np.timedelta64:
        return np.timedelta64(int(self.params["utc_offset_hours"]), "h")


def probs(d: dict[Any, float]) -> tuple[list[Any], list[float]]:
    keys = list(d.keys())
    vals = [float(d[k]) for k in keys]
    return keys, vals


def to_ts(arr: np.ndarray) -> np.ndarray:
    """Truncate a datetime64 array to whole seconds."""
    return arr.astype("datetime64[s]")


def random_time_of_day(
    rng: np.random.Generator, n: int, hour_weights: list[float] | None = None
) -> np.ndarray:
    """Offsets (timedelta64[s]) within a day, optionally following an hourly weight profile."""
    if hour_weights is None:
        secs = rng.integers(0, 86_400, size=n)
    else:
        w = np.asarray(hour_weights, dtype=float)
        hours = rng.choice(24, size=n, p=w / w.sum())
        secs = hours * 3600 + rng.integers(0, 3600, size=n)
    return secs.astype("timedelta64[s]")


def month_index(dates: np.ndarray, start: np.datetime64) -> np.ndarray:
    m = dates.astype("datetime64[M]").astype(int) - start.astype("datetime64[M]").astype(int)
    return m
