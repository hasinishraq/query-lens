"""Semantic layer: the 24 official metrics in ``semantic/metrics.yaml``."""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache

import yaml
from pydantic import BaseModel

from .paths import SEMANTIC

PERIOD_START = date(2024, 1, 1)
PERIOD_END = date(2026, 10, 31)


class Metric(BaseModel):
    name: str
    description: str
    grain: str
    tables: list[str]
    excludes: str = ""
    notes: str = ""
    sql: str


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Metric]:
    with open(SEMANTIC / "metrics.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)["metrics"]
    return {name: Metric(name=name, **body) for name, body in raw.items()}


_PARAM = re.compile(r":(start_date|end_date)\b")


def render(sql: str, start: date = PERIOD_START, end: date = PERIOD_END) -> str:
    """Replace ``:start_date`` / ``:end_date`` with DATE literals (values are dates, never user text)."""
    values = {"start_date": f"DATE '{start.isoformat()}'", "end_date": f"DATE '{end.isoformat()}'"}
    return _PARAM.sub(lambda m: values[m.group(1)], sql)
