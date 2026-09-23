"""Retry progress reaches the UI, recovered failures stay hidden, and
Anthropic thinking survives into the next request."""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx

from delta_app.conversation import CodingSession
from delta_app.tui.adapter import RetryNotice, SessionBridge
from delta_app.ui.render import TranscriptRenderer
from delta_harness.contracts.stream import MessageEndEvent, RetryEvent, RunEndEvent
from delta_harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ModelEntry,
    TextSegment,
    ThoughtSegment,
    ToolOutcomeEntry,
)
from delta_harness.driver import RuntimeConfig, RuntimeHarness
from delta_harness.provider.wire import SourceRetryEvent, StreamCloseEvent, StreamFaultEvent
from delta_model.claude import AnthropicProvider, _compile_messages
from delta_model.oai_compatible import OpenAIProvider
from delta_model.scripted import ReplayProvider
from delta_model.settings import AnthropicProfile, Credential, OpenAIProfile, RetryPolicy

# ---------------------------------------------------------------------------
# Provider retries are announced
# ---------------------------------------------------------------------------

_CHAT_OK = "".join(
    f"data: {json.dumps(f)}\n\n"
    for f in (
        {"id": "c", "choices": [{"index": 0, "delta": {"content": "hi"}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
) + "data: [DONE]\n\n"


def _openai(responses: list[httpx.Response]) -> OpenAIProvider:
    provider = OpenAIProvider(OpenAIProfile(
        name="openai",
        credential=Credential(api_key="k", base_url="https://api.openai.com/v1"),
        retry=RetryPolicy(max_retries=2, max_delay_seconds=0.01),
    ))
    queue = list(responses)
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: queue.pop(0)),
    )
    return provider


async def test_http_retry_is_announced_before_backoff() -> None:
    provider = _openai([httpx.Response(503, text="busy"), httpx.Response(200, text=_CHAT_OK)])
    events = [e async for e in provider.stream_response(
        model="gpt-4o", system="s", messages=[HumanEntry(content="q")], tools=[],
    )]
    retries = [e for e in events if isinstance(e, SourceRetryEvent)]
    assert len(retries) == 1
    assert retries[0].attempt == 2 and retries[0].max_attempts == 3
    assert "HTTP 503" in retries[0].message
    assert isinstance(events[-1], StreamCloseEvent)


async def test_retry_becomes_an_agent_event_and_is_rendered() -> None:
    provider = _openai([httpx.Response(429, text="slow down"), httpx.Response(200, text=_CHAT_OK)])
    harness = RuntimeHarness(RuntimeConfig(provider=provider, model="gpt-4o", system="s"))
    events = [e async for e in harness.submit("q")]
    retries = [e for e in events if isinstance(e, RetryEvent)]
    assert len(retries) == 1

    out = io.StringIO()
    renderer = TranscriptRenderer(out)
    for event in events:
        renderer.render(event)
    assert "[retry] Reattempting request 2/3 after HTTP 429" in out.getvalue()


# ---------------------------------------------------------------------------
# Session recoveries hide the failure they replace
# ---------------------------------------------------------------------------

_OVERFLOW = "prompt is too long: 250000 tokens > 200000 maximum"


def _ok(text: str) -> list:
    return [StreamCloseEvent(reason="stop", message=ModelEntry(
        content=[TextSegment(text=text)], stop_reason="stop"))]


def _overflow() -> list:
    return [StreamFaultEvent(reason="error", error=ModelEntry(
        stop_reason="error", error_message=_OVERFLOW))]


async def _session(tmp_path: Path, streams: list) -> CodingSession:
    session = await CodingSession.create(
        provider=ReplayProvider(streams), provider_name="test", model="m", system="s",
        sessions_dir=tmp_path,
    )

    async def no_title() -> str:
        return ""

    session.auto_name = no_title  # type: ignore[method-assign]
    return session


def _errors(events: list) -> list:
    return [e for e in events if isinstance(e, MessageEndEvent)
            and isinstance(e.message, ModelEntry) and e.message.error_message]


async def test_recovered_overflow_shows_progress_not_the_error(tmp_path: Path) -> None:
    streams = [_ok(f"r{i}") for i in range(4)] + [_overflow(), _ok("summary"), _ok("done")]
    session = await _session(tmp_path, streams)
    for i in range(4):
        _ = [e async for e in session.submit(f"p{i}")]

    events = [e async for e in session.submit("big")]
    assert not _errors(events)
    assert [e.message for e in events if isinstance(e, RetryEvent)] == [
        "Context window exceeded; compacted older history and retrying."
    ]
    assert sum(isinstance(e, RunEndEvent) for e in events) == 1


async def test_unrecovered_overflow_still_shows_the_error(tmp_path: Path) -> None:
    session = await _session(tmp_path, [_overflow()])
    events = [e async for e in session.submit("big")]
    assert [e.message.error_message for e in _errors(events)] == [_OVERFLOW]
    assert not any(isinstance(e, RetryEvent) for e in events)


async def test_tui_shows_retry_as_a_notice(tmp_path: Path) -> None:
    streams = [_ok(f"r{i}") for i in range(4)] + [_overflow(), _ok("summary"), _ok("done")]
    session = await _session(tmp_path, streams)
    for i in range(4):
        _ = [e async for e in session.submit(f"p{i}")]
    bridge = SessionBridge(session)
    updates = [u async for u in bridge.submit("big")]
    notices = [u for u in updates if isinstance(u, RetryNotice)]
    assert len(notices) == 1
    assert not any(getattr(u, "error", None) for u in updates)


# ---------------------------------------------------------------------------
# Anthropic thinking: signatures and redacted blocks are kept and replayed
# ---------------------------------------------------------------------------


def _sse(*events: tuple[str, dict]) -> str:
    return "".join(f"event: {n}\ndata: {json.dumps(p)}\n\n" for n, p in events)


_THINKING_TURN = _sse(
    ("message_start", {"message": {"id": "m1", "model": "claude", "usage": {}}}),
    ("content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
    ("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "Plan."}}),
    ("content_block_delta", {"index": 0, "delta": {"type": "signature_delta", "signature": "SIG-A"}}),
    ("content_block_stop", {"index": 0}),
    ("content_block_start", {"index": 1, "content_block": {"type": "redacted_thinking",
                                                            "data": "ENCRYPTED"}}),
    ("content_block_stop", {"index": 1}),
    ("content_block_start", {"index": 2, "content_block": {"type": "tool_use", "id": "toolu_1",
                                                            "name": "Read"}}),
    ("content_block_delta", {"index": 2, "delta": {"type": "input_json_delta",
                                                   "partial_json": '{"p": 1}'}}),
    ("content_block_stop", {"index": 2}),
    ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
    ("message_stop", {}),
)


async def test_anthropic_thinking_round_trips_into_the_next_request() -> None:
    provider = AnthropicProvider(AnthropicProfile(
        name="anthropic",
        credential=Credential(api_key="k", base_url="https://api.anthropic.com"),
        retry=RetryPolicy(max_retries=0),
    ))
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=_THINKING_TURN)),
    )
    events = [e async for e in provider.stream_response(
        model="claude", system="s", messages=[HumanEntry(content="q")], tools=[],
    )]
    await provider.close()
    reply = events[-1].message
    thoughts = [s for s in reply.content if isinstance(s, ThoughtSegment)]
    assert [(t.thinking, t.redacted, t.thinking_signature) for t in thoughts] == [
        ("Plan.", False, "SIG-A"), ("", True, "ENCRYPTED"),
    ]

    history = [
        HumanEntry(content="q"), reply,
        ToolOutcomeEntry(tool_call_id="toolu_1", tool_name="Read", content="r"),
    ]
    assistant = _compile_messages(history)[1]["content"]
    assert assistant[0] == {"type": "thinking", "thinking": "Plan.", "signature": "SIG-A"}
    assert assistant[1] == {"type": "redacted_thinking", "data": "ENCRYPTED"}
    assert assistant[2]["type"] == "tool_use"


def test_unsigned_thinking_is_still_not_sent() -> None:
    reply = ModelEntry(api="messages", content=[
        ThoughtSegment(thinking="no proof"), CallBlock(id="t1", name="Read", arguments={}),
    ])
    blocks = _compile_messages([HumanEntry(content="q"), reply])[1]["content"]
    assert [b["type"] for b in blocks] == ["tool_use"]
