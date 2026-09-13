"""End-to-end tests for OpenAIProvider.

These drive the *whole* provider composition — transport -> dispatch.decode_stream
-> parsers -> normalize -> WireEvents — against a mocked HTTP transport that
returns canned Server-Sent Events. No network. This is the coverage the
per-module unit tests (test_openai_provider.py) do not provide: they test the
pieces; this tests that the pieces are wired together correctly.
"""

from __future__ import annotations

import json

import httpx

from delta_harness.contracts.transcript import HumanEntry
from delta_harness.provider.wire import (
    CallCloseEvent,
    ContentChunkEvent,
    ContentCloseEvent,
    StreamCloseEvent,
    StreamFaultEvent,
    StreamOpenEvent,
)
from delta_model.oai_compatible import OpenAIProvider
from delta_model.settings import Credential, OpenAIProfile, RetryPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile(max_retries: int = 0) -> OpenAIProfile:
    return OpenAIProfile(
        name="openai",
        credential=Credential(api_key="test-key", base_url="https://api.openai.com/v1"),
        retry=RetryPolicy(max_retries=max_retries),
    )


def _sse(*frames: str) -> str:
    """Encode JSON frames as an SSE body (blank line between frames)."""
    return "".join(f"data: {frame}\n\n" for frame in frames)


def _provider_with(handler) -> OpenAIProvider:
    """Build a provider whose HTTP client is backed by a MockTransport."""
    provider = OpenAIProvider(_profile())
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


async def _collect(provider: OpenAIProvider, **kwargs) -> list:
    events = [event async for event in provider.stream_response(**kwargs)]
    await provider.close()
    return events


_ASK = {"model": "gpt-4o", "system": "sys", "messages": [HumanEntry(content="hi")], "tools": []}


# ---------------------------------------------------------------------------
# Text streaming
# ---------------------------------------------------------------------------


async def test_text_stream_produces_full_wire_sequence() -> None:
    body = _sse(
        json.dumps({
            "id": "c1", "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }),
        json.dumps({
            "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
        }),
        json.dumps({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }),
        "[DONE]",
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


async def test_request_targets_chat_completions_endpoint() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, text=_sse(
            json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            "[DONE]",
        ))

    await _collect(_provider_with(handler), **_ASK)

    assert seen["url"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer test-key"


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


async def test_streamed_tool_call_reassembles_arguments() -> None:
    body = _sse(
        json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "t1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"ci'}}
        ]}, "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'ty":"NYC"}'}}
        ]}, "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        "[DONE]",
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


async def test_malformed_sse_json_becomes_fault() -> None:
    body = _sse('{"choices": [{"index": 0, "delta": {"content": "hi"}}]}', "{not valid json")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    events = await _collect(_provider_with(handler), **_ASK)

    assert any(isinstance(e, StreamFaultEvent) for e in events)
