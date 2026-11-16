"""Pydantic config models and loaders for ``configs/*.yaml``."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from .paths import CONFIGS


class Limits(BaseModel):
    rpm: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    tpd: int | None = None
    tokens_per_month: int | None = None


class ModelSpec(BaseModel):
    id: str
    family: str | None = None
    native_tools: bool = True  # False -> JSON-in-text tool protocol
    limits: Limits = Field(default_factory=Limits)


class Provider(BaseModel):
    name: str = ""
    base_url: str
    base_url_env: str | None = None
    key_env: str | None
    family: str
    daily_reset_tz: str = "UTC"
    supports_seed: bool = True  # Gemini's OpenAI-compatible endpoint rejects `seed` (HTTP 400)
    account_limits: Limits | None = None
    models: list[ModelSpec]

    def resolved_base_url(self) -> str:
        if self.base_url_env and os.environ.get(self.base_url_env):
            return os.environ[self.base_url_env]
        return self.base_url

    def api_key(self) -> str | None:
        if self.key_env is None:
            return "ollama"
        return os.environ.get(self.key_env) or None

    @property
    def enabled(self) -> bool:
        return self.api_key() is not None

    def model(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"model {model_id!r} is not configured for provider {self.name!r}")

    def family_of(self, model_id: str) -> str:
        try:
            return self.model(model_id).family or self.family
        except KeyError:
            return self.family


class ModelRef(BaseModel):
    provider: str
    model: str


class Budgets(BaseModel):
    max_tool_calls: int = 15
    max_llm_calls: int = 6
    max_seconds: float = 120
    max_input_tokens: int = 20_000
    tool_output_tokens: int = 1_200


class Params(BaseModel):
    temperature: float = 0.0
    seed: int | None = 42
    max_tokens: int = 1200


class Cascade(ModelRef):
    escalate_on: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    name: str
    agent: ModelRef
    fallbacks: list[ModelRef] = Field(default_factory=list)
    cascade: Cascade | None = None
    judge: ModelRef
    judge_fallbacks: list[ModelRef] = Field(default_factory=list)
    prompt: str = "system_v1.md"
    params: Params = Field(default_factory=Params)
    budgets: Budgets = Field(default_factory=Budgets)
    gate: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _judge_independent(self) -> AgentConfig:
        """Rule 14: the judge must not be from the agent's model family."""
        providers = load_providers()
        agent_fams = {providers[self.agent.provider].family_of(self.agent.model)}
        if self.cascade:
            agent_fams.add(providers[self.cascade.provider].family_of(self.cascade.model))
        for j in [self.judge, *self.judge_fallbacks]:
            fam = providers[j.provider].family_of(j.model)
            if fam in agent_fams:
                raise ValueError(
                    f"judge {j.provider}/{j.model} is in model family {fam!r}, same as the agent (rule 14)"
                )
        return self


class SqlRuntime(BaseModel):
    timeout_s: float = 10
    memory_limit: str = "1GB"
    max_rows: int = 1000
    explain_row_threshold: int = 200_000
    threads: int = 2


class RetrieverRuntime(BaseModel):
    default_k: int = 10
    bm25_weight: float = 0.5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class LlmRuntime(BaseModel):
    cache_dir: str = ".cache/llm"
    max_retries: int = 4
    backoff_base_s: float = 1.5
    backoff_max_s: float = 60
    failures_before_fallback: int = 3
    request_timeout_s: float = 90
    quota_db: str = "data/quota.sqlite"


class AgentRuntime(BaseModel):
    repeat_call_limit: int = 2
    easy_item_max_tool_calls: int = 12


class PromptRuntime(BaseModel):
    max_system_tokens: int = 1500


class Runtime(BaseModel):
    sql: SqlRuntime = Field(default_factory=SqlRuntime)
    retriever: RetrieverRuntime = Field(default_factory=RetrieverRuntime)
    llm: LlmRuntime = Field(default_factory=LlmRuntime)
    agent: AgentRuntime = Field(default_factory=AgentRuntime)
    prompt: PromptRuntime = Field(default_factory=PromptRuntime)


def _read(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_providers() -> dict[str, Provider]:
    raw = _read(CONFIGS / "providers.yaml")["providers"]
    return {name: Provider(name=name, **body) for name, body in raw.items()}


@lru_cache(maxsize=1)
def load_runtime() -> Runtime:
    return Runtime(**_read(CONFIGS / "runtime.yaml"))


def load_config(name_or_path: str) -> AgentConfig:
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIGS / f"{name_or_path}.yaml"
    return AgentConfig(**_read(p))


@lru_cache(maxsize=1)
def load_prices() -> dict[str, dict[str, float]]:
    return _read(CONFIGS / "prices.yaml").get("prices", {})


def shadow_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p = load_prices().get(model)
    if not p:
        return 0.0
    return (tokens_in * p["input"] + tokens_out * p["output"]) / 1e6
