"""End-to-end tests for AnthropicProvider.

These drive the *whole* provider composition — courier (SSE transport) ->
emitter (ResponseMachine) -> WireEvents — against a mocked HTTP transport that
returns canned Anthropic Messages-API events. No network. The per-module unit
tests (test_anthropic_provider.py) cover the pieces in isolation; this covers
that they are wired together correctly.
"""

from __future__ import annotations

import json

import httpx

from delta_harness.contracts.transcript import HumanEntry
from delta_harness.provider.wire import (
    CallCloseEvent,
    ContentChunkEvent,
    ContentCloseEvent,
    ReasoningChunkEvent,
    ReasoningCloseEvent,
    StreamCloseEvent,
    StreamFaultEvent,
    StreamOpenEvent,
)
from delta_model.anthropic import AnthropicProvider
from delta_model.settings import AnthropicProfile, Credential, RetryPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile(max_retries: int = 0) -> AnthropicProfile:
    return AnthropicProfile(
        name="anthropic",
        credential=Credential(api_key="test-key", base_url="https://api.anthropic.com"),
        retry=RetryPolicy(max_retries=max_retries),
    )


def _sse(*events: tuple[str, dict]) -> str:
    """Encode ``(event_name, payload)`` pairs as an Anthropic SSE body."""
    return "".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events
    )


_START = ("message_start", {
    "message": {
        "id": "msg_1", "model": "claude-sonnet-4-20250514",
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
})


def _provider_with(handler) -> AnthropicProvider:
    provider = AnthropicProvider(_profile())
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


async def _collect(provider: AnthropicProvider, **kwargs) -> list:
    events = [event async for event in provider.stream_response(**kwargs)]
    await provider.close()
    return events


_ASK = {"model": "claude-sonnet-4-20250514", "system": "sys",
        "messages": [HumanEntry(content="hi")], "tools": []}


# ---------------------------------------------------------------------------
# Text streaming
# ---------------------------------------------------------------------------


async def test_text_stream_produces_full_wire_sequence() -> None:
    body = _sse(
        _START,
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": " world"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
        ("message_stop", {}),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    events = await _collect(_provider_with(handler), **_ASK)

    assert isinstance(events[0], StreamOpenEvent)
    assert isinstance(events[-1], StreamCloseEvent)
    assert events[-1].reason == "stop"

    text = "".join(e.delta for e in events if isinstance(e, ContentChunkEvent))
    assert text == "Hello world"

    closes = [e for e in events if isinstance(e, ContentCloseEvent)]
    assert closes and closes[0].content == "Hello world"

    final = events[-1].message
    assert final.usage.input == 10
    assert final.usage.output == 5


async def test_request_targets_messages_endpoint_with_headers() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key", "")
        seen["version"] = request.headers.get("anthropic-version", "")
        return httpx.Response(200, text=_sse(
            _START,
            ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
            ("message_stop", {}),
        ))

    await _collect(_provider_with(handler), **_ASK)

    assert seen["url"].endswith("/v1/messages")
    assert seen["key"] == "test-key"
    assert seen["version"]  # anthropic-version header is present


# ---------------------------------------------------------------------------
# Thinking + tool calls
# ---------------------------------------------------------------------------


async def test_thinking_blocks_stream_as_reasoning() -> None:
    body = _sse(
        _START,
        ("content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
        ("message_stop", {}),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    events = await _collect(_provider_with(handler), **_ASK)

    reasoning = "".join(e.delta for e in events if isinstance(e, ReasoningChunkEvent))
    assert reasoning == "Let me think"
    closes = [e for e in events if isinstance(e, ReasoningCloseEvent)]
    assert closes and closes[0].content == "Let me think"


async def test_streamed_tool_call_reassembles_arguments() -> None:
    body = _sse(
        _START,
        ("content_block_start", {"index": 0, "content_block": {
            "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"ci'}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": 'ty":"NYC"}'}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
        ("message_stop", {}),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    events = await _collect(_provider_with(handler), **_ASK)

    assert events[-1].reason == "toolUse"
    call_closes = [e for e in events if isinstance(e, CallCloseEvent)]
    assert len(call_closes) == 1
    assert call_closes[0].tool_call.name == "get_weather"
    assert call_closes[0].tool_call.arguments == {"city": "NYC"}


# ---------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------


async def test_http_error_yields_single_fault_event() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    events = await _collect(_provider_with(handler), **_ASK)

    assert len(events) == 1
    assert isinstance(events[0], StreamFaultEvent)
    assert events[0].reason == "error"
    assert "bad request" in (events[0].error.error_message or "")
