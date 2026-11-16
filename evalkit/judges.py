"""LLM judges J1-J3 (CLAUDE.md 8.5).

``Judge(rubric, examples).grade(trace) -> Verdict``. Temperature 0, critique before verdict, structured
JSON output, few-shot examples from the **train** split only, and a model family different from the agent's
(rule 14, enforced in :class:`querylens.config.AgentConfig` and again here).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from querylens.config import ModelRef, Params, load_providers
from querylens.llm import LLMClient

from .schemas import Verdict

ROOT = Path(__file__).resolve().parents[1]
RUBRICS = ROOT / "rubrics"
JUDGES = {"j1": "j1_ambiguity.md", "j2": "j2_faithfulness.md", "j3": "j3_refusal.md"}
APPLIES_TO = {
    "j1": {"ambiguous", "golden"},
    "j2": {"golden", "bilingual_bn", "bilingual_banglish"},
    "j3": {"adversarial"},
}

JUDGE_SYSTEM = (
    "You are an evaluation judge. Read the rubric and the trace. Write a short critique first, then the "
    'verdict. Reply with ONLY a JSON object: {"critique": "<2-4 sentences>", "verdict": "PASS" or "FAIL"}.'
)


@dataclass
class JudgeExample:
    trace_summary: str
    critique: str
    verdict: str
    split: str = "train"


def trace_summary(trace: dict[str, Any], max_chars: int = 6000) -> str:
    """Compact, judge-readable view of one agent run."""
    parts = [f"QUESTION: {trace.get('question', '')}", f"ROLE: {trace.get('role', '')}"]
    for s in trace.get("steps", []):
        parts.append(f"- {s}")
    if trace.get("clarifying_question"):
        parts.append(f"AGENT ASKED: {trace['clarifying_question']}")
    if trace.get("final_sql"):
        parts.append(f"FINAL SQL: {trace['final_sql']}")
    if trace.get("result_rows") is not None:
        parts.append(f"RESULT ROWS (from the database): {json.dumps(trace['result_rows'][:30], default=str)}")
    parts.append(f"FINAL ANSWER: {trace.get('final_text', '')}")
    text = "\n".join(parts)
    return text[:max_chars]


@dataclass
class Judge:
    name: str
    rubric: str
    chain: list[ModelRef]
    examples: list[JudgeExample] = field(default_factory=list)
    agent_family: str | None = None
    client: LLMClient | None = None

    def __post_init__(self) -> None:
        for ex in self.examples:
            if ex.split != "train":
                raise ValueError("judge few-shot examples must come from the train split (rule 3)")
        if self.agent_family:
            provs = load_providers()
            for ref in self.chain:
                if provs[ref.provider].family_of(ref.model) == self.agent_family:
                    raise ValueError(f"judge {ref.model} shares the agent's model family (rule 14)")

    @classmethod
    def load(
        cls,
        name: str,
        chain: list[ModelRef],
        examples: list[JudgeExample] | None = None,
        agent_family: str | None = None,
        client: LLMClient | None = None,
    ) -> Judge:
        rubric = (RUBRICS / JUDGES[name]).read_text(encoding="utf-8")
        return cls(
            name=name,
            rubric=rubric,
            chain=chain,
            examples=examples or [],
            agent_family=agent_family,
            client=client,
        )

    @property
    def rubric_hash(self) -> str:
        return hashlib.sha256(self.rubric.encode("utf-8")).hexdigest()[:12]

    def messages(self, trace: dict[str, Any]) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": JUDGE_SYSTEM + "\n\nRUBRIC:\n" + self.rubric}]
        for ex in self.examples:
            msgs.append({"role": "user", "content": ex.trace_summary})
            msgs.append(
                {"role": "assistant", "content": json.dumps({"critique": ex.critique, "verdict": ex.verdict})}
            )
        msgs.append({"role": "user", "content": trace_summary(trace)})
        return msgs

    def grade(self, trace: dict[str, Any]) -> Verdict:
        client = self.client or LLMClient()
        resp = client.complete(
            self.messages(trace), None, self.chain, Params(temperature=0, seed=42, max_tokens=400)
        )
        return parse_verdict(resp.content or "")


def parse_verdict(text: str) -> Verdict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            v = Verdict.model_validate_json(m.group(0))
            return Verdict(critique=v.critique, verdict=v.verdict.upper())  # type: ignore[arg-type]
        except ValidationError:
            pass
    up = text.upper()
    verdict = "FAIL" if "FAIL" in up and ("PASS" not in up or up.rfind("FAIL") > up.rfind("PASS")) else "PASS"
    return Verdict(critique=f"[unparsed judge output] {text[:300]}", verdict=verdict)  # type: ignore[arg-type]


def write_lock(judges: list[Judge], path: Path | None = None) -> None:
    """Freeze judges: rubric hash, model chain, params (``rubrics/judges.lock.json``)."""
    lock = {
        j.name: {
            "rubric": JUDGES[j.name],
            "rubric_hash": j.rubric_hash,
            "models": [r.model_dump() for r in j.chain],
            "params": {"temperature": 0, "seed": 42},
            "n_examples": len(j.examples),
        }
        for j in judges
    }
    (path or RUBRICS / "judges.lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
