"""Pydantic models for everything evalkit reads or writes (CLAUDE.md 8.1, 8.9)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Split = Literal["train", "dev", "test", "adv"]
SetName = Literal[
    "golden", "ambiguous", "unanswerable", "adversarial", "bilingual_bn", "bilingual_banglish", "regression"
]
Intent = Literal[
    "lookup",
    "aggregation",
    "ranking",
    "trend",
    "comparison",
    "ratio",
    "cohort",
    "funnel",
    "clarify",
    "abstain",
    "attack",
]
Difficulty = Literal["easy", "medium", "hard", "extra_hard"]
Trap = Literal[
    "none",
    "status_spelling",
    "soft_delete",
    "payment_retries",
    "scd2",
    "timezone",
    "currency",
    "nulls",
    "duplicate_names",
]
Ambiguity = Literal["clear", "metric", "entity", "period"]
Persona = Literal["terse", "vague", "non_native", "banglish"]
Adversarial = Literal["none", "injection", "pii", "write", "resource", "jailbreak"]
ExpectedType = Literal["result", "clarify", "abstain", "refuse"]

TRAPS: tuple[str, ...] = (
    "status_spelling",
    "soft_delete",
    "payment_retries",
    "scd2",
    "timezone",
    "currency",
    "nulls",
    "duplicate_names",
)
DOMAINS: tuple[str, ...] = (
    "orders",
    "payments",
    "logistics",
    "customers",
    "catalog",
    "sellers",
    "inventory",
    "marketing",
    "web",
    "support",
    "reference",
)


class Dims(BaseModel):
    intent: Intent
    difficulty: Difficulty
    domains: list[str]
    trap: Trap = "none"
    metric: str | None = None
    ambiguity: Ambiguity = "clear"
    answerable: bool = True
    persona: Persona = "terse"
    adversarial: Adversarial = "none"


class Expected(BaseModel):
    type: ExpectedType = "result"
    float_tol: float = 0.0005
    ordered: bool = False
    must_mention_any: list[str] = Field(default_factory=list)
    forbidden_columns: list[str] = Field(default_factory=list)


class EvalItem(BaseModel):
    id: str
    split: Split
    set: SetName
    role: str = "analyst"
    lang: Literal["en", "bn", "banglish"] = "en"
    question: str
    dims: Dims
    gold_sql_file: str | None = None
    gold_tables: list[str] = Field(default_factory=list)
    expected: Expected = Field(default_factory=Expected)
    pair_id: str | None = None
    planted_fact: str | None = None
    verified_by: str | None = None
    notes: str | None = None


class Verdict(BaseModel):
    critique: str
    verdict: Literal["PASS", "FAIL"]


class CheckResult(BaseModel):
    mode: str  # F1..F11 or 'exec'
    passed: bool
    detail: str = ""
    applicable: bool = True


class ItemResult(BaseModel):
    run_id: str
    item_id: str
    repeat: int
    split: str
    set: str
    dims: dict[str, Any]
    expected_type: str
    trace_id: str
    status: str
    behaviour: str  # answered | clarified | abstained | refused | error
    exec_match: bool | None = None  # result items only
    pred_fingerprint: list[Any] | None = None
    gold_fingerprint: list[Any] | None = None
    checks: list[CheckResult] = Field(default_factory=list)
    judges: dict[str, Verdict] = Field(default_factory=dict)
    unsafe_access: bool = False
    tool_calls: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    shadow_cost_usd: float = 0.0
    models_used: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    first_search_tables: list[str] = Field(default_factory=list)
    final_tables: list[str] = Field(default_factory=list)
    verified: bool = False
    finished_at: str = ""

    @property
    def passed(self) -> bool:
        if self.expected_type == "result":
            return bool(self.exec_match)
        return (
            self.behaviour
            == {"clarify": "clarified", "abstain": "abstained", "refuse": "refused"}[self.expected_type]
        )


class RunInfo(BaseModel):
    run_id: str
    config: str
    git_sha: str
    snapshot: str
    slice: str
    repeats: int
    model: str
    prompt_hash: str
    dataset_hash: str
    started: str
    item_ids: list[str]
    judges: bool = False
