"""Statistics for evals (CLAUDE.md 8.6). Pure functions, unit-tested."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

Z95 = 1.959963984540054


def standard_error(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def wilson_ci(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def rogan_gladen(p_obs: float, tpr: float, tnr: float) -> float:
    """Corrected prevalence (p + TNR - 1) / (TPR + TNR - 1), clipped to [0, 1]."""
    denom = tpr + tnr - 1
    if denom <= 0:
        return float("nan")
    return float(min(1.0, max(0.0, (p_obs + tnr - 1) / denom)))


def rogan_gladen_ci(
    judge_fail: Sequence[bool],
    labels_human_fail: Sequence[bool],
    labels_judge_fail: Sequence[bool],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Corrected failure prevalence with a bootstrap CI that resamples both the unlabelled judge verdicts
    and the labelled set used to estimate TPR/TNR (FAIL = positive class)."""
    jf = np.asarray(judge_fail, dtype=bool)
    hf = np.asarray(labels_human_fail, dtype=bool)
    lj = np.asarray(labels_judge_fail, dtype=bool)

    def tpr_tnr(h: np.ndarray, j: np.ndarray) -> tuple[float, float]:
        pos, neg = h, ~h
        tpr = (j & pos).sum() / max(pos.sum(), 1)
        tnr = (~j & neg).sum() / max(neg.sum(), 1)
        return float(tpr), float(tnr)

    tpr, tnr = tpr_tnr(hf, lj)
    point = rogan_gladen(float(jf.mean()), tpr, tnr)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        a = jf[rng.integers(0, len(jf), len(jf))]
        k = rng.integers(0, len(hf), len(hf))
        t1, t2 = tpr_tnr(hf[k], lj[k])
        v = rogan_gladen(float(a.mean()), t1, t2)
        if not math.isnan(v):
            out.append(v)
    if not out:
        return point, (float("nan"), float("nan"))
    return point, (float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2)))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k = 1 - C(n-c, k) / C(n, k)."""
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k = C(c, k) / C(n, k): probability that k independent tries all pass."""
    if k > n:
        raise ValueError("k must be <= n")
    return math.comb(c, k) / math.comb(n, k)


@dataclass
class PairedResult:
    fixed: int
    broken: int
    unchanged_pass: int
    unchanged_fail: int
    p_value: float

    @property
    def n(self) -> int:
        return self.fixed + self.broken + self.unchanged_pass + self.unchanged_fail

    @property
    def net(self) -> int:
        return self.fixed - self.broken


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b (fixed) and c (broken)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sps.binom.cdf(min(b, c), n, 0.5)))


def paired_comparison(before: dict[str, bool], after: dict[str, bool]) -> PairedResult:
    keys = sorted(set(before) & set(after))
    fixed = sum(1 for k in keys if not before[k] and after[k])
    broken = sum(1 for k in keys if before[k] and not after[k])
    up = sum(1 for k in keys if before[k] and after[k])
    uf = sum(1 for k in keys if not before[k] and not after[k])
    return PairedResult(fixed, broken, up, uf, mcnemar_exact(fixed, broken))


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    if len(x) == 0:
        return float("nan")
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)
