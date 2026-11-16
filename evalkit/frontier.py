"""Accuracy / tokens / latency frontier across configs (Phase 11). Reads finished runs in results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PUBLISHED = RESULTS / "published"


def collect(slice_name: str, size: str) -> list[dict]:
    out = []
    for p in sorted(RESULTS.glob("*/summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if (
            s.get("slice") == slice_name
            and s.get("snapshot") == size
            and (s.get("exec_accuracy") or {}).get("rate") is not None
        ):
            out.append(s)
    latest: dict[str, dict] = {}
    for s in out:  # newest run per config
        latest[s["config"]] = s
    return list(latest.values())


def plot_frontier(
    slice_name: str, size: str, run_configs: list[str] | None = None, limit: int | None = None
) -> Path:
    if run_configs:
        from .runner import run

        for c in run_configs:
            run(slice_name, size, 1, c, limit=limit)
    runs = collect(slice_name, size)
    PUBLISHED.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for s in runs:
        x = s["components"]["tokens_in_mean"]
        y = s["exec_accuracy"]["rate"]
        lo, hi = s["exec_accuracy"]["ci95"]
        p95 = s["components"]["latency_all"]["p95"]
        cost = s["components"].get("shadow_cost_usd_total", 0.0)
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="o", capsize=3)
        ax.annotate(
            f"{s['config']}\np95 {p95:.1f}s, ${cost:.3f} shadow",
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
        rows.append((s["config"], y, lo, hi, x, p95, cost, s["run_id"]))
    ax.set_xlabel("input tokens per question (mean)")
    ax.set_ylabel("execution accuracy")
    ax.set_title(f"Accuracy vs tokens ({slice_name}, {size}); API spend $0")
    ax.grid(alpha=0.3)
    out = PUBLISHED / f"frontier_{slice_name}_{size}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    md = [
        "| Config | Exec acc | 95% CI | Tokens in / q | p95 latency (s) | Shadow cost ($) | Run |",
        "|---|---|---|---|---|---|---|",
    ]
    md += [
        f"| {c} | {y:.3f} | [{lo:.3f}, {hi:.3f}] | {x:.0f} | {p:.1f} | {k:.4f} | {r} |"
        for c, y, lo, hi, x, p, k, r in rows
    ]
    (PUBLISHED / f"frontier_{slice_name}_{size}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out
