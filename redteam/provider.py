"""promptfoo Python provider: sends a red-team prompt to the QueryLens agent and returns its answer.

Config via env: QUERYLENS_RT_CONFIG (default b_mistral), QUERYLENS_RT_SIZE (default full), QUERYLENS_RT_ROLE.
The returned metadata carries guard rejections so assertions can check unsafe access attempts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

_AGENT = None


def _agent():
    global _AGENT
    if _AGENT is None:
        from dotenv import load_dotenv

        from querylens.agent import Agent
        from querylens.config import load_config

        load_dotenv(ROOT / ".env")
        _AGENT = Agent(
            load_config(os.environ.get("QUERYLENS_RT_CONFIG", "b_mistral")),
            role=os.environ.get("QUERYLENS_RT_ROLE", "analyst"),
            size=os.environ.get("QUERYLENS_RT_SIZE", "full"),
        )
    return _AGENT


def call_api(prompt: str, options: dict, context: dict) -> dict:
    from evalkit.checks import UNSAFE_RULES

    res = _agent().run(prompt, scenario_id="redteam", run_id="redteam")
    rules = res.guard_rejections()
    unsafe = [r for r in rules if r in UNSAFE_RULES]
    text = res.final_text or res.clarifying_question or f"[{res.status}: {res.stop_reason}]"
    return {
        "output": text,
        "metadata": {
            "trace_id": res.trace_id,
            "guard_rejections": rules,
            "unsafe_attempts": unsafe,
            "final_sql": res.final_sql,
        },
    }
