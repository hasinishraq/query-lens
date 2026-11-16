"""Hash-based deterministic randomness (CLAUDE.md section 6.4).

Every random draw in the generator comes from a generator returned by :func:`rng_for`, seeded from a
hash of ``(seed, table, purpose)``. Adding a new column (a new purpose) never changes the stream used
by existing columns. Never use the global ``random`` module or unseeded NumPy generators.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def stable_hash(*parts: object) -> int:
    """64-bit integer hash of the string forms of ``parts`` (stable across runs and platforms)."""
    h = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def rng_for(seed: int, table: str, purpose: str) -> np.random.Generator:
    """A NumPy generator dedicated to one (table, purpose) pair."""
    return np.random.default_rng(np.random.SeedSequence(stable_hash(seed, table, purpose)))


def allocate(total: int, weights: Sequence[float] | np.ndarray) -> np.ndarray:
    """Split ``total`` into integer parts proportional to ``weights`` (largest remainder; deterministic)."""
    w = np.asarray(weights, dtype=float)
    if total < 0:
        raise ValueError("total must be >= 0")
    if w.sum() <= 0:
        raise ValueError("weights must have a positive sum")
    raw = w / w.sum() * total
    base = np.floor(raw).astype(np.int64)
    rem = int(total - base.sum())
    if rem:
        # ties broken by index so the result is deterministic
        order = np.lexsort((np.arange(len(w)), -(raw - base)))
        base[order[:rem]] += 1
    return base


def choice(rng: np.random.Generator, values: Sequence[object], probs: Sequence[float], n: int) -> np.ndarray:
    """``n`` draws from ``values`` with probabilities ``probs`` (normalised)."""
    p = np.asarray(probs, dtype=float)
    return rng.choice(np.asarray(values, dtype=object), size=n, p=p / p.sum())


def exact_choice(
    rng: np.random.Generator, values: Sequence[object], probs: Sequence[float], n: int
) -> np.ndarray:
    """``n`` values whose counts match ``probs`` exactly (largest remainder), in shuffled order."""
    counts = allocate(n, probs)
    out = np.repeat(np.asarray(values, dtype=object), counts)
    rng.shuffle(out)
    return out


def pick_k(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """Sorted indices of ``k`` distinct items out of ``n``."""
    if k > n:
        raise ValueError(f"cannot pick {k} of {n}")
    return np.sort(rng.choice(n, size=k, replace=False))
