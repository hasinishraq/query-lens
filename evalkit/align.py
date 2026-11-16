"""Judge validation against human labels: confusion matrix, TPR, TNR, Cohen's kappa (FAIL = positive)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .stats import cohens_kappa


@dataclass
class Alignment:
    n: int
    tp: int
    fn: int
    fp: int
    tn: int
    tpr: float
    tnr: float
    kappa: float
    accuracy: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def meets(self, target: float = 0.85) -> bool:
        return self.tpr >= target and self.tnr >= target


def align(human: Sequence[str], judge: Sequence[str]) -> Alignment:
    """``human`` and ``judge`` are 'PASS'/'FAIL' lists for the same items."""
    if len(human) != len(judge):
        raise ValueError("label lists differ in length")
    h = [x.upper() == "FAIL" for x in human]
    j = [x.upper() == "FAIL" for x in judge]
    tp = sum(1 for a, b in zip(h, j, strict=True) if a and b)
    fn = sum(1 for a, b in zip(h, j, strict=True) if a and not b)
    fp = sum(1 for a, b in zip(h, j, strict=True) if not a and b)
    tn = sum(1 for a, b in zip(h, j, strict=True) if not a and not b)
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    n = len(h)
    return Alignment(
        n=n,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        tpr=tpr,
        tnr=tnr,
        kappa=cohens_kappa(h, j),
        accuracy=(tp + tn) / n if n else float("nan"),
    )
