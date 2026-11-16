"""``querylens`` CLI: ask one question, check free models, show quota usage."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from .config import load_config, load_providers, load_runtime
from .paths import ROOT


def cmd_ask(a: argparse.Namespace) -> int:
    from .agent import Agent

    cfg = load_config(a.config)
    agent = Agent(cfg, role=a.role, size=a.size)
    try:
        res = agent.run(a.question)
    finally:
        agent.close()
    print(f"trace_id: {res.trace_id}")
    print(f"status:   {res.status} ({res.stop_reason})")
    print(f"models:   {', '.join(res.models_used) or '-'}  fallback={res.fallback_used}")
    print(
        f"calls:    {res.n_llm_calls} LLM, {res.n_tool_calls} tools; tokens in/out {res.tokens_in}/{res.tokens_out}; "
        f"{res.latency_s:.1f}s"
    )
    if res.clarifying_question:
        print(f"\nQUESTION: {res.clarifying_question}")
    if res.final_text:
        print(f"\nANSWER:\n{res.final_text}")
    if res.final_sql:
        print(f"\nSQL:\n{res.final_sql}")
    return 0 if res.status in ("answered", "clarified") else 1


def cmd_models_check(a: argparse.Namespace) -> int:
    """List models each enabled provider serves now; fail if a configured model is gone."""
    import openai

    rt = load_runtime()
    bad = 0
    for name, p in load_providers().items():
        if not p.enabled:
            print(f"[skip] {name}: no key in ${p.key_env}")
            continue
        try:
            client = openai.OpenAI(
                base_url=p.resolved_base_url(),
                api_key=p.api_key(),
                timeout=rt.llm.request_timeout_s,
                max_retries=1,
            )
            served = {m.id for m in client.models.list().data}
        except Exception as e:  # noqa: BLE001 - report any provider failure and continue
            print(f"[fail] {name}: could not list models ({type(e).__name__}: {str(e)[:120]})")
            bad += 1
            continue
        for m in p.models:
            ok = m.id in served or m.id.removeprefix("models/") in {s.removeprefix("models/") for s in served}
            print(f"[{'ok' if ok else 'GONE'}] {name}/{m.id}")
            bad += 0 if ok else 1
        if a.verbose:
            for s in sorted(served):
                print(f"       serves: {s}")
    return 1 if bad else 0


def cmd_quota(a: argparse.Namespace) -> int:
    from .ratelimit import RateLimiter

    rt = load_runtime()
    lim = RateLimiter(ROOT / rt.llm.quota_db, load_providers())
    print(
        f"{'provider/model':45} {'req today':>10} {'rpd':>8} {'tokens today':>13} {'tpd':>10} {'tokens month':>13}"
    )
    for u in lim.usage_today():
        print(
            f"{u.provider + '/' + u.model:45} {u.requests_today:>10} {str(u.limits.rpd or '-'):>8} "
            f"{u.tokens_today:>13} {str(u.limits.tpd or '-'):>10} {u.tokens_month:>13}"
        )
    return 0


def main(argv: list[str] | None = None) -> None:
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="querylens")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ask")
    s.add_argument("question")
    s.add_argument("--role", default="analyst")
    s.add_argument("--size", default="full")
    s.add_argument("--config", default="b_mistral")
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("models-check")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_models_check)
    s = sub.add_parser("quota")
    s.set_defaults(fn=cmd_quota)
    a = ap.parse_args(argv)
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()


__all__ = ["main", "json"]
