"""Cross-provider history replay and in-stream failure recovery."""

from __future__ import annotations

import json
import re

import httpx

from delta_harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ModelEntry,
    ThoughtSegment,
    ToolOutcomeEntry,
)
from delta_harness.provider.wire import StreamCloseEvent, StreamFaultEvent, StreamOpenEvent
from delta_model._oai.payloads import build_chat_payload, build_responses_payload
from delta_model.claude import AnthropicProvider, _compile_messages
from delta_model.correlation import wire_call_id
from delta_model.oai_compatible import OpenAIProvider
from delta_model.settings import AnthropicProfile, Credential, OpenAIProfile, RetryPolicy

_ANTHROPIC_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# ---------------------------------------------------------------------------
# Tool-call id portability
# ---------------------------------------------------------------------------


def test_safe_ids_pass_through_unchanged() -> None:
    assert wire_call_id("toolu_01ABC-x") == "toolu_01ABC-x"


def test_unsafe_ids_hash_deterministically() -> None:
    raw = "call_XwFnGCtoQNIN2ID9ahtpLvsI|fc_0ca1b059695ff22f"
    first, second = wire_call_id(raw), wire_call_id(raw)
    assert first == second
    assert first.startswith("tc_") and len(first) == 43
    assert _ANTHROPIC_ID.fullmatch(first)


def test_distinct_unsafe_ids_never_collide() -> None:
    assert wire_call_id("a|b") != wire_call_id("a_b")
    assert wire_call_id("a|b") != wire_call_id("a.b")


def test_overlong_ids_are_hashed() -> None:
    assert wire_call_id("x" * 65).startswith("tc_")


def _foreign_history() -> list:
    """History minted by a Responses-style vendor, with piped ids and reasoning."""
    ids = ["call_one|fc_1", "call_two|fc_2"]
    return [
        HumanEntry(content="go"),
        ModelEntry(api="responses", content=[
            ThoughtSegment(thinking="plan", thinking_signature="enc_opaque"),
            CallBlock(id=ids[0], name="read", arguments={}),
            CallBlock(id=ids[1], name="read", arguments={}),
        ]),
        ToolOutcomeEntry(tool_call_id=ids[0], tool_name="read", content="a"),
        ToolOutcomeEntry(tool_call_id=ids[1], tool_name="read", content="b"),
        ModelEntry(api="responses", content=[
            ThoughtSegment(thinking="only thinking", thinking_signature="enc_2"),
        ]),
        HumanEntry(content="next"),
    ]


def test_anthropic_replay_of_foreign_history_is_valid() -> None:
    messages = _compile_messages(_foreign_history())
    blocks = [b for m in messages for b in m["content"]]

    uses = [b["id"] for b in blocks if b["type"] == "tool_use"]
    results = [b["tool_use_id"] for b in blocks if b["type"] == "tool_result"]
    assert uses == results
    assert len(set(uses)) == 2
    assert all(_ANTHROPIC_ID.fullmatch(i) for i in uses)
    assert not any(b["type"] == "thinking" for b in blocks)
    # The reasoning-only turn vanishes instead of becoming empty content.
    assert all(m["content"] for m in messages)
    roles = [m["role"] for m in messages]
    assert all(a != b for a, b in zip(roles, roles[1:], strict=False))


def test_native_anthropic_reasoning_survives() -> None:
    history = [
        HumanEntry(content="q"),
        ModelEntry(api="messages", content=[
            ThoughtSegment(thinking="t", thinking_signature="sig"),
            CallBlock(id="toolu_1", name="read", arguments={}),
        ]),
        ToolOutcomeEntry(tool_call_id="toolu_1", tool_name="read", content="r"),
    ]
    blocks = [b for m in _compile_messages(history) for b in m["content"]]
    assert blocks[1]["type"] == "thinking"
    assert any(b.get("id") == "toolu_1" for b in blocks)


def test_openai_payloads_translate_ids_consistently() -> None:
    history = _foreign_history()
    chat = build_chat_payload(model="m", system="s", messages=history, tools=[])
    call_ids = [c["id"] for m in chat["messages"] for c in m.get("tool_calls", [])]
    result_ids = [m["tool_call_id"] for m in chat["messages"] if m["role"] == "tool"]
    assert call_ids == result_ids
    assert all(_ANTHROPIC_ID.fullmatch(i) for i in call_ids)

    resp = build_responses_payload(model="m", system="s", messages=history, tools=[])
    calls = [i["call_id"] for i in resp["input"] if i.get("type") == "function_call"]
    outs = [i["call_id"] for i in resp["input"] if i.get("type") == "function_call_output"]
    assert calls == outs == call_ids


# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------


def _mock(bodies: list[str]) -> tuple[httpx.AsyncClient, list[int]]:
    """A client that serves ``bodies`` in order, repeating the last one."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, text=bodies[min(len(calls), len(bodies)) - 1])

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


async def _drain(provider) -> list:
    events = [e async for e in provider.stream_response(
        model="m", system="s", messages=[HumanEntry(content="q")], tools=[],
    )]
    await provider.close()
    return events


# ---------------------------------------------------------------------------
# Anthropic in-stream retry
# ---------------------------------------------------------------------------


def _anthropic_sse(*events: tuple[str, dict]) -> str:
    return "".join(f"event: {n}\ndata: {json.dumps(p)}\n\n" for n, p in events)


_A_START = ("message_start", {"message": {"id": "m", "model": "c", "usage": {}}})
_A_TEXT = [
    ("content_block_start", {"index": 0, "content_block": {"type": "text"}}),
    ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
    ("content_block_stop", {"index": 0}),
    ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
    ("message_stop", {}),
]


def _a_error(kind: str = "overloaded_error") -> tuple[str, dict]:
    return ("error", {"type": "error", "error": {"type": kind, "message": "Overloaded"}})


def _anthropic(bodies: list[str], retries: int = 2) -> tuple[AnthropicProvider, list[int]]:
    provider = AnthropicProvider(AnthropicProfile(
        name="anthropic",
        credential=Credential(api_key="k", base_url="https://api.anthropic.com"),
        retry=RetryPolicy(max_retries=retries, max_delay_seconds=0.01),
    ))
    provider._client, calls = _mock(bodies)
    return provider, calls


async def test_anthropic_transient_error_before_content_is_retried_quietly() -> None:
    provider, calls = _anthropic([
        _anthropic_sse(_A_START, _a_error()),
        _anthropic_sse(_A_START, *_A_TEXT),
    ])
    events = await _drain(provider)
    assert len(calls) == 2
    assert sum(isinstance(e, StreamOpenEvent) for e in events) == 1
    assert not any(isinstance(e, StreamFaultEvent) for e in events)
    assert isinstance(events[-1], StreamCloseEvent)
    assert events[-1].message.text == "hi"


async def test_anthropic_error_after_content_is_terminal() -> None:
    provider, calls = _anthropic([_anthropic_sse(_A_START, *_A_TEXT[:2], _a_error())])
    events = await _drain(provider)
    assert len(calls) == 1
    assert isinstance(events[-1], StreamFaultEvent)


async def test_anthropic_non_transient_error_is_terminal() -> None:
    provider, calls = _anthropic([_anthropic_sse(_A_START, _a_error("authentication_error"))])
    events = await _drain(provider)
    assert len(calls) == 1
    assert isinstance(events[-1], StreamFaultEvent)


async def test_anthropic_persistent_overload_surfaces_after_budget() -> None:
    provider, calls = _anthropic([_anthropic_sse(_A_START, _a_error())], retries=2)
    events = await _drain(provider)
    assert len(calls) == 3
    assert isinstance(events[-1], StreamFaultEvent)
    assert events[-1].error.error_message == "Overloaded"


# ---------------------------------------------------------------------------
# OpenAI-compatible in-stream retry
# ---------------------------------------------------------------------------


def _chat_sse(*frames: dict) -> str:
    return "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"


_CHAT_OK = [
    {"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]},
    {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
]
_OVERLOAD = {"error": {"type": "service_unavailable_error",
                       "code": "server_is_overloaded",
                       "message": "Our servers are currently overloaded."}}


def _openai(bodies: list[str], retries: int = 2) -> tuple[OpenAIProvider, list[int]]:
    provider = OpenAIProvider(OpenAIProfile(
        name="openai",
        credential=Credential(api_key="k", base_url="https://api.openai.com/v1"),
        retry=RetryPolicy(max_retries=retries, max_delay_seconds=0.01),
    ))
    provider._client, calls = _mock(bodies)
    return provider, calls


async def test_openai_transient_error_before_content_is_retried() -> None:
    provider, calls = _openai([_chat_sse(_OVERLOAD), _chat_sse(*_CHAT_OK)])
    events = await _drain(provider)
    assert len(calls) == 2
    assert not any(isinstance(e, StreamFaultEvent) for e in events)
    assert events[-1].message.text == "hi"


async def test_openai_quota_exhaustion_is_not_retried() -> None:
    quota = {"error": {"code": "rate_limit_exceeded",
                       "message": "insufficient_quota: check your billing"}}
    provider, calls = _openai([_chat_sse(quota)])
    events = await _drain(provider)
    assert len(calls) == 1
    assert isinstance(events[-1], StreamFaultEvent)


async def test_openai_terminal_error_shows_nested_message() -> None:
    provider, _calls = _openai([_chat_sse(_OVERLOAD)], retries=0)
    events = await _drain(provider)
    assert events[-1].error.error_message == "Our servers are currently overloaded."


async def test_openai_error_after_content_is_terminal() -> None:
    provider, calls = _openai([_chat_sse(_CHAT_OK[0], _OVERLOAD)])
    events = await _drain(provider)
    assert len(calls) == 1
    assert isinstance(events[-1], StreamFaultEvent)


async def test_responses_failed_event_reads_response_error() -> None:
    failed = {"type": "response.failed", "response": {"error": {
        "code": "server_error", "message": "Upstream blew up"}}}
    body = f"event: response.failed\ndata: {json.dumps(failed)}\n\n"
    provider = OpenAIProvider(OpenAIProfile(
        name="openai",
        credential=Credential(api_key="k", base_url="https://api.openai.com/v1"),
        retry=RetryPolicy(max_retries=0),
    ))
    provider._client, _calls = _mock([body])
    events = [e async for e in provider.stream_response(
        model="gpt-5-codex", system="s", messages=[HumanEntry(content="q")], tools=[],
    )]
    await provider.close()
    assert events[-1].error.error_message == "Upstream blew up"
