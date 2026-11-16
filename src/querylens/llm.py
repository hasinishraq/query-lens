"""One OpenAI-compatible client for every free provider (CLAUDE.md 7.4).

``LLMClient.complete(messages, tools, chain)`` picks the provider from the config chain, applies the rate
limiter, retries 429/5xx with exponential backoff + jitter (honouring ``retry-after``), moves to the next
model after repeated failures, caches replies on disk and records the model id *returned by the API*.

Modes (env ``QUERYLENS_LLM_MODE``):
- ``live`` (default): cache read-through; network on miss.
- ``replay``: cache only; a miss raises :class:`CacheMiss` (tests and CI).
- ``refresh``: ignore the cache, call the API, overwrite the cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import openai
from pydantic import BaseModel, Field, ValidationError

from .config import ModelRef, Params, Provider, load_providers, load_runtime
from .paths import ROOT
from .ratelimit import QuotaExhausted, RateLimiter

log = logging.getLogger(__name__)


class CacheMiss(RuntimeError):
    pass


class AllModelsFailed(RuntimeError):
    pass


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = ""


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    provider: str
    requested_model: str
    returned_model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cached: bool = False
    fallback_used: bool = False
    finish_reason: str | None = None
    attempts: int = 1


def estimate_tokens(obj: Any) -> int:
    """Rough token count (4 characters per token) used for budgets before the API reports usage."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


def cache_key(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    params: dict[str, Any],
) -> str:
    blob = json.dumps(
        {"p": provider, "m": model, "msg": messages, "t": tools or [], "params": params},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)


# --------------------------------------------------------------------------- JSON-in-text tool protocol
class _TextToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


TEXT_TOOL_INSTRUCTIONS = (
    "You cannot call tools natively. To call a tool, reply with ONLY one JSON object on a single line:\n"
    '{"tool": "<tool name>", "arguments": {...}}\n'
    "Available tools (JSON schema):\n"
)


def tools_as_text(tools: list[dict[str, Any]]) -> str:
    lines = [TEXT_TOOL_INSTRUCTIONS]
    for t in tools:
        f = t["function"]
        lines.append(
            json.dumps(
                {
                    "name": f["name"],
                    "description": f.get("description", ""),
                    "parameters": f.get("parameters", {}),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_text_tool_call(text: str) -> ToolCall | None:
    """Parse the JSON-in-text protocol; returns None when the reply holds no valid tool call."""
    if not text:
        return None
    candidates = [text.strip()]
    m = _JSON_OBJ.search(text)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        c = c.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            obj = _TextToolCall.model_validate_json(c)
        except ValidationError:
            continue
        return ToolCall(
            id="text-0",
            name=obj.tool,
            arguments=obj.arguments,
            raw_arguments=json.dumps(obj.arguments, ensure_ascii=False),
        )
    return None


# --------------------------------------------------------------------------- client
class LLMClient:
    def __init__(
        self,
        providers: dict[str, Provider] | None = None,
        limiter: RateLimiter | None = None,
        cache: DiskCache | None = None,
        mode: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        client_factory: Callable[[Provider], Any] | None = None,
    ) -> None:
        rt = load_runtime().llm
        self.rt = rt
        self.providers = providers or load_providers()
        self.limiter = limiter or RateLimiter(ROOT / rt.quota_db, self.providers)
        self.cache = cache or DiskCache(ROOT / rt.cache_dir)
        self.mode = mode or os.environ.get("QUERYLENS_LLM_MODE", "live")
        self.sleep = sleep
        self._clients: dict[str, Any] = {}
        self._factory = client_factory or self._default_factory

    def _default_factory(self, p: Provider) -> Any:
        return openai.OpenAI(
            base_url=p.resolved_base_url(),
            api_key=p.api_key() or "missing",
            timeout=self.rt.request_timeout_s,
            max_retries=0,
        )

    def _client(self, provider: str) -> Any:
        if provider not in self._clients:
            self._clients[provider] = self._factory(self.providers[provider])
        return self._clients[provider]

    def available(self, ref: ModelRef) -> bool:
        p = self.providers.get(ref.provider)
        return p is not None and p.enabled

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chain: list[ModelRef],
        params: Params | None = None,
    ) -> LLMResponse:
        params = params or Params()
        errors: list[str] = []
        for i, ref in enumerate(chain):
            if self.mode != "replay" and not self.available(ref):
                errors.append(f"{ref.provider}: no API key ({self.providers[ref.provider].key_env})")
                continue
            try:
                resp = self._complete_one(messages, tools, ref, params)
                resp.fallback_used = i > 0
                return resp
            except QuotaExhausted as e:
                errors.append(str(e))
                log.warning("quota: %s; trying next model", e)
            except CacheMiss:
                raise
            except _GiveUp as e:
                errors.append(str(e))
                log.warning("giving up on %s/%s: %s", ref.provider, ref.model, e)
        raise AllModelsFailed("; ".join(errors) or "empty model chain")

    def _native_tools(self, ref: ModelRef) -> bool:
        try:
            return self.providers[ref.provider].model(ref.model).native_tools
        except KeyError:
            return True

    def _complete_one(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        ref: ModelRef,
        params: Params,
    ) -> LLMResponse:
        native = self._native_tools(ref)
        send_msgs = messages
        send_tools = tools
        if tools and not native:
            send_msgs = [{"role": "system", "content": tools_as_text(tools)}, *messages]
            send_tools = None
        p = {"temperature": params.temperature, "max_tokens": params.max_tokens}
        if params.seed is not None:
            p["seed"] = params.seed
        key = cache_key(ref.provider, ref.model, send_msgs, send_tools, p)
        if self.mode != "refresh":
            hit = self.cache.get(key)
            if hit is not None:
                r = LLMResponse(**hit)
                r.cached = True
                r.latency_s = 0.0
                return r
        if self.mode == "replay":
            raise CacheMiss(f"no cached reply for {ref.provider}/{ref.model} (key {key[:12]})")
        est = estimate_tokens(send_msgs) + estimate_tokens(send_tools or []) + params.max_tokens
        failures = 0
        attempt = 0
        while True:
            attempt += 1
            row = self.limiter.acquire(ref.provider, ref.model, est)
            t0 = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {"model": ref.model, "messages": send_msgs, **p}
                if send_tools:
                    kwargs["tools"] = send_tools
                    kwargs["tool_choice"] = "auto"
                raw = self._client(ref.provider).chat.completions.create(**kwargs)
            except (
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APIConnectionError,
                openai.APITimeoutError,
            ) as e:
                self.limiter.record(row, 0)
                failures += 1
                if failures >= self.rt.failures_before_fallback or attempt > self.rt.max_retries:
                    raise _GiveUp(f"{type(e).__name__} after {attempt} attempts") from e
                self.sleep(self._backoff(attempt, e))
                continue
            except openai.APIStatusError as e:
                self.limiter.record(row, 0)
                raise _GiveUp(f"HTTP {e.status_code}: {str(e)[:200]}") from e
            latency = time.perf_counter() - t0
            resp = self._parse(raw, ref, native, latency, attempt)
            self.limiter.record(row, resp.tokens_in + resp.tokens_out)
            self.cache.put(key, resp.model_dump())
            return resp

    def _backoff(self, attempt: int, err: Exception) -> float:
        retry_after = None
        resp = getattr(err, "response", None)
        if resp is not None:
            ra = resp.headers.get("retry-after")
            if ra:
                try:
                    retry_after = float(ra)
                except ValueError:
                    retry_after = None
        if retry_after is not None:
            return min(retry_after, self.rt.backoff_max_s)
        base = self.rt.backoff_base_s * (2 ** (attempt - 1))
        return min(self.rt.backoff_max_s, base * (0.5 + random.random()))  # noqa: S311 (jitter only)

    @staticmethod
    def _parse(raw: Any, ref: ModelRef, native: bool, latency: float, attempt: int) -> LLMResponse:
        choice = raw.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            args_raw = tc.function.arguments or "{}"
            try:
                args = json.loads(args_raw)
                if not isinstance(args, dict):
                    args = {"_value": args}
            except json.JSONDecodeError:
                args = {"_invalid_json": args_raw}
            calls.append(
                ToolCall(
                    id=tc.id or f"call-{len(calls)}",
                    name=tc.function.name,
                    arguments=args,
                    raw_arguments=args_raw,
                )
            )
        content = msg.content
        if not native and not calls:
            tc = parse_text_tool_call(content or "")
            if tc:
                calls.append(tc)
        usage = getattr(raw, "usage", None)
        return LLMResponse(
            content=content,
            tool_calls=calls,
            provider=ref.provider,
            requested_model=ref.model,
            returned_model=getattr(raw, "model", None) or ref.model,
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_s=round(latency, 3),
            finish_reason=getattr(choice, "finish_reason", None),
            attempts=attempt,
        )


class _GiveUp(RuntimeError):
    pass
