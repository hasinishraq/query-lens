"""LLM client against a fake OpenAI-compatible server: 429, retry-after, fallback, cache, replay, text tools."""

import pytest

from querylens.config import Limits, ModelRef, ModelSpec, Params, Provider
from querylens.llm import (
    AllModelsFailed,
    CacheMiss,
    DiskCache,
    LLMClient,
    parse_text_tool_call,
    tools_as_text,
)
from querylens.ratelimit import RateLimiter
from tests.fake_openai import FakeServer, chat_reply

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "x",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]
MSGS = [{"role": "user", "content": "hi"}]


def _providers(url_a: str, url_b: str | None = None, native: bool = True) -> dict[str, Provider]:
    p = {
        "a": Provider(
            name="a",
            base_url=url_a,
            key_env=None,
            family="fam-a",
            models=[ModelSpec(id="m1", native_tools=native, limits=Limits(rpm=1000))],
        )
    }
    if url_b:
        p["b"] = Provider(
            name="b",
            base_url=url_b,
            key_env=None,
            family="fam-b",
            models=[ModelSpec(id="m2", limits=Limits(rpm=1000))],
        )
    return p


def _client(tmp_path, providers, mode="live"):
    sleeps: list[float] = []
    c = LLMClient(
        providers=providers,
        limiter=RateLimiter(tmp_path / "q.sqlite", providers, sleep=sleeps.append),
        cache=DiskCache(tmp_path / "cache"),
        mode=mode,
        sleep=sleeps.append,
    )
    return c, sleeps


def test_tool_call_and_returned_model(tmp_path):
    with FakeServer() as s:
        s.push(
            chat_reply(
                tool_calls=[{"name": "run_sql", "arguments": {"query": "SELECT 1"}}], model="m1-2026-03"
            )
        )
        c, _ = _client(tmp_path, _providers(s.url))
        r = c.complete(MSGS, TOOLS, [ModelRef(provider="a", model="m1")], Params())
    assert r.tool_calls[0].name == "run_sql"
    assert r.tool_calls[0].arguments == {"query": "SELECT 1"}
    assert r.returned_model == "m1-2026-03" and r.requested_model == "m1"
    assert (r.tokens_in, r.tokens_out) == (100, 20)
    assert s.requests[0]["temperature"] == 0 and s.requests[0]["seed"] == 42


def test_429_honours_retry_after(tmp_path):
    with FakeServer() as s:
        s.push({"error": {"message": "rate"}}, status=429, headers={"retry-after": "7"})
        s.push(chat_reply("ok"))
        c, sleeps = _client(tmp_path, _providers(s.url))
        r = c.complete(MSGS, None, [ModelRef(provider="a", model="m1")])
    assert r.content == "ok" and r.attempts == 2
    assert 7.0 in sleeps


def test_5xx_then_fallback(tmp_path):
    with FakeServer() as a, FakeServer() as b:
        for _ in range(3):
            a.push({"error": {"message": "boom"}}, status=500)
        b.push(chat_reply("from b", model="m2"))
        c, sleeps = _client(tmp_path, _providers(a.url, b.url))
        r = c.complete(MSGS, None, [ModelRef(provider="a", model="m1"), ModelRef(provider="b", model="m2")])
    assert r.content == "from b"
    assert r.fallback_used is True and r.provider == "b"
    assert len(sleeps) == 2  # exponential backoff between the failed attempts


def test_all_models_fail(tmp_path):
    with FakeServer() as a:
        for _ in range(3):
            a.push({"error": {"message": "boom"}}, status=503)
        c, _ = _client(tmp_path, _providers(a.url))
        with pytest.raises(AllModelsFailed):
            c.complete(MSGS, None, [ModelRef(provider="a", model="m1")])


def test_cache_hit_avoids_network(tmp_path):
    with FakeServer() as s:
        s.push(chat_reply("first"))
        c, _ = _client(tmp_path, _providers(s.url))
        r1 = c.complete(MSGS, None, [ModelRef(provider="a", model="m1")])
        r2 = c.complete(MSGS, None, [ModelRef(provider="a", model="m1")])
    assert r1.content == r2.content == "first"
    assert r2.cached and len(s.requests) == 1


def test_replay_mode_cache_miss(tmp_path):
    c, _ = _client(tmp_path, _providers("http://127.0.0.1:9/v1"), mode="replay")
    with pytest.raises(CacheMiss):
        c.complete(MSGS, None, [ModelRef(provider="a", model="m1")])


def test_json_in_text_protocol(tmp_path):
    with FakeServer() as s:
        s.push(chat_reply('Sure. {"tool": "run_sql", "arguments": {"query": "SELECT 2"}}'))
        c, _ = _client(tmp_path, _providers(s.url, native=False))
        r = c.complete(MSGS, TOOLS, [ModelRef(provider="a", model="m1")])
    assert r.tool_calls[0].name == "run_sql" and r.tool_calls[0].arguments["query"] == "SELECT 2"
    assert "tools" not in s.requests[0]
    assert "cannot call tools natively" in s.requests[0]["messages"][0]["content"]


def test_parse_text_tool_call_variants():
    assert (
        parse_text_tool_call('```json\n{"tool": "ask_user", "arguments": {"question": "Which?"}}\n```').name
        == "ask_user"
    )
    assert parse_text_tool_call("no tool here") is None
    assert parse_text_tool_call('{"tool": 5}') is None
    assert "run_sql" in tools_as_text(TOOLS)


def test_quota_exhausted_moves_to_next_model(tmp_path):
    with FakeServer() as a, FakeServer() as b:
        b.push(chat_reply("b"))
        provs = _providers(a.url, b.url)
        provs["a"].models[0].limits = Limits(rpd=0)
        c, _ = _client(tmp_path, provs)
        r = c.complete(MSGS, None, [ModelRef(provider="a", model="m1"), ModelRef(provider="b", model="m2")])
    assert r.provider == "b" and r.fallback_used
    assert len(a.requests) == 0
