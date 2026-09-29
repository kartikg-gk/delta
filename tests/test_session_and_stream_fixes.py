"""Regression tests: retry codes, proxy schemes, model choice on rewind and
resume, slash input in the terminal UI, and Responses stream edge cases."""

from __future__ import annotations

import json
import ssl
from pathlib import Path

import httpx
import pytest

from delta_app.conversation import CodingSession
from delta_harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ModelEntry,
    TextSegment,
    ThoughtSegment,
)
from delta_harness.provider.wire import StreamCloseEvent
from delta_harness.session.index import SessionCatalog
from delta_model._claude.courier import ApiRejection
from delta_model._oai.transport import HttpStreamError
from delta_model.oai_compatible import OpenAIProvider
from delta_model.scripted import ReplayProvider
from delta_model.settings import Credential, OpenAIProfile, RetryPolicy
from delta_model.transport.client import build_async_client

# ---------------------------------------------------------------------------
# Retry status codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 501, 504, 505, 529, 599])
def test_transient_statuses_are_retried(status: int) -> None:
    assert HttpStreamError(status, "").retriable
    assert ApiRejection(status, "").retriable


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_client_errors_are_not_retried(status: int) -> None:
    assert not HttpStreamError(status, "").retriable
    assert not ApiRejection(status, "").retriable


# ---------------------------------------------------------------------------
# Generic SOCKS proxy URLs
# ---------------------------------------------------------------------------


async def test_generic_socks_proxy_builds_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:1080")
    client = build_async_client()
    await client.aclose()


# ---------------------------------------------------------------------------
# Model choice on rewind and resume
# ---------------------------------------------------------------------------


def _ok(text: str) -> list:
    return [StreamCloseEvent(reason="stop", message=ModelEntry(
        content=[TextSegment(text=text)], stop_reason="stop"))]


async def _quiet(session: CodingSession) -> CodingSession:
    async def no_title() -> str:
        return ""

    session.auto_name = no_title  # type: ignore[method-assign]
    return session


async def test_rewind_keeps_the_active_model(tmp_path: Path) -> None:
    session = await _quiet(await CodingSession.create(
        provider=ReplayProvider([_ok("a"), _ok("b")]), provider_name="openai",
        model="old-model", system="s", sessions_dir=tmp_path,
    ))
    _ = [e async for e in session.submit("one")]
    point = session._record_ids[-1]
    await session.switch_model("new-model")
    _ = [e async for e in session.submit("two")]
    await session.rewind(point)
    assert session.model == "new-model"


async def _session_on(tmp_path: Path, provider_name: str, model: str) -> str:
    session = await _quiet(await CodingSession.create(
        provider=ReplayProvider([_ok("a")]), provider_name=provider_name,
        model=model, system="s", sessions_dir=tmp_path,
    ))
    _ = [e async for e in session.submit("hi")]
    await session.shutdown()
    return session.session_id


async def test_resume_under_the_same_provider_restores_the_session_model(tmp_path: Path) -> None:
    sid = await _session_on(tmp_path, "openai", "gpt-a")
    resumed = await CodingSession.resume(
        sid, provider=ReplayProvider([]), provider_name="openai", model="gpt-default",
        system="s", sessions_dir=tmp_path,
    )
    assert resumed.model == "gpt-a"


async def test_resume_under_another_provider_keeps_the_active_model(tmp_path: Path) -> None:
    sid = await _session_on(tmp_path, "anthropic", "claude-x")
    resumed = await CodingSession.resume(
        sid, provider=ReplayProvider([]), provider_name="openai", model="gpt-default",
        system="s", sessions_dir=tmp_path,
    )
    assert resumed.model == "gpt-default"


async def test_resume_with_an_explicit_model_uses_it(tmp_path: Path) -> None:
    sid = await _session_on(tmp_path, "openai", "gpt-a")
    resumed = await CodingSession.resume(
        sid, provider=ReplayProvider([]), provider_name="openai", model="gpt-chosen",
        system="s", sessions_dir=tmp_path, keep_model=True,
    )
    assert resumed.model == "gpt-chosen"


async def test_provider_switch_is_recorded_for_resume(tmp_path: Path) -> None:
    session = await _quiet(await CodingSession.create(
        provider=ReplayProvider([_ok("a")]), provider_name="openai",
        model="gpt-a", system="s", sessions_dir=tmp_path,
    ))
    _ = [e async for e in session.submit("hi")]
    await session.switch_provider(ReplayProvider([]), "anthropic", "claude-x")
    meta = SessionCatalog(tmp_path).get(session.session_id)
    assert meta is not None and meta.provider == "anthropic"


# ---------------------------------------------------------------------------
# Slash input in the terminal UI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "command"), [
    ("/model", True),
    ("/q", True),
    ("/compact", True),
    ("/tmp/shot.png what is this", False),
    ("/Users/me/notes.md summarise", False),
    ("/skill:review src", False),
    ("/my-template args", False),
    ("/nosuchcmd hello", False),
])
def test_only_registered_commands_are_run_as_commands(text: str, command: bool) -> None:
    pytest.importorskip("textual")
    from delta_app.tui.app import _is_registered_command

    assert _is_registered_command(text) is command


# ---------------------------------------------------------------------------
# Responses stream edge cases
# ---------------------------------------------------------------------------


def _frames(*frames: dict, named: bool = False) -> str:
    out = ""
    for frame in frames:
        if named:
            out += f"event: {frame['type']}\n"
        out += f"data: {json.dumps(frame)}\n\n"
    return out


async def _responses(body: str) -> ModelEntry:
    provider = OpenAIProvider(OpenAIProfile(
        name="openai",
        credential=Credential(api_key="k", base_url="https://api.openai.com/v1"),
        retry=RetryPolicy(max_retries=0),
    ))
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body)),
    )
    events = [e async for e in provider.stream_response(
        model="o4-mini", system="s", messages=[HumanEntry(content="q")], tools=[],
    )]
    await provider.close()
    return events[-1].message


_ITEM = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "Read", "arguments": ""}
_DONE = {"type": "response.completed", "response": {"status": "completed"}}


def _calls(reply: ModelEntry) -> list[tuple[str, dict]]:
    return [(c.name, c.arguments) for c in reply.content if isinstance(c, CallBlock)]


async def test_events_without_an_event_line_are_read_from_their_type() -> None:
    reply = await _responses(_frames(
        {"type": "response.output_text.delta", "delta": "hello"}, _DONE,
    ))
    assert reply.text == "hello"


async def test_empty_final_arguments_keep_the_streamed_ones() -> None:
    reply = await _responses(_frames(
        {"type": "response.output_item.added", "output_index": 0, "item": _ITEM},
        {"type": "response.function_call_arguments.delta", "output_index": 0,
         "delta": '{"path": "a"}'},
        {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": ""},
        _DONE, named=True,
    ))
    assert _calls(reply) == [("Read", {"path": "a"})]


async def test_non_empty_final_arguments_still_win() -> None:
    reply = await _responses(_frames(
        {"type": "response.output_item.added", "output_index": 0, "item": _ITEM},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{}"},
        {"type": "response.function_call_arguments.done", "output_index": 0,
         "arguments": '{"path": "b"}'},
        _DONE, named=True,
    ))
    assert _calls(reply) == [("Read", {"path": "b"})]


async def test_call_finished_only_by_its_item_is_kept_once() -> None:
    reply = await _responses(_frames(
        {"type": "response.output_item.added", "output_index": 0, "item": _ITEM},
        {"type": "response.function_call_arguments.delta", "output_index": 0,
         "delta": '{"path": "c"}'},
        {"type": "response.output_item.done", "output_index": 0,
         "item": {**_ITEM, "arguments": '{"path": "c"}'}},
        _DONE, named=True,
    ))
    assert _calls(reply) == [("Read", {"path": "c"})]


async def test_reasoning_summary_parts_are_separate_paragraphs() -> None:
    reply = await _responses(_frames(
        {"type": "response.reasoning_summary_text.delta", "delta": "First"},
        {"type": "response.reasoning_summary_part.done"},
        {"type": "response.reasoning_summary_text.delta", "delta": "Second"},
        {"type": "response.reasoning_summary_part.done"},
        {"type": "response.output_text.delta", "delta": "Done"},
        _DONE, named=True,
    ))
    thoughts = [s.thinking for s in reply.content if isinstance(s, ThoughtSegment)]
    assert thoughts and thoughts[0].startswith("First\n\nSecond")


# ---------------------------------------------------------------------------
# Gateway cost and provider labels
# ---------------------------------------------------------------------------


def test_reported_gateway_cost_is_recorded() -> None:
    from delta_model._oai.helpers import extract_usage

    usage = extract_usage({
        "prompt_tokens": 8, "completion_tokens": 10, "total_tokens": 18, "cost": 0.0000192,
        "cost_details": {"upstream_inference_prompt_cost": 0.0000032,
                         "upstream_inference_completions_cost": 0.000016},
    })
    assert usage.cost.total == pytest.approx(0.0000192)
    assert usage.cost.input == pytest.approx(0.0000032)
    assert usage.cost.output == pytest.approx(0.000016)
    assert extract_usage({"prompt_tokens": 1, "completion_tokens": 1}).cost.total == 0.0


def test_compatible_providers_are_labelled_by_their_own_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from delta_app.config.config import resolve_provider

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    provider = resolve_provider("openrouter")
    assert provider._profile.name == "openrouter"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Transport and TLS failures
# ---------------------------------------------------------------------------

_CHAT_OK = (
    'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def _chat_provider(handler) -> OpenAIProvider:  # type: ignore[no-untyped-def]
    provider = OpenAIProvider(OpenAIProfile(
        name="openai",
        credential=Credential(api_key="k", base_url="https://api.openai.com/v1"),
        retry=RetryPolicy(max_retries=2, max_delay_seconds=0.01),
    ))
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


@pytest.mark.parametrize("error", [
    ssl.SSLError(1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac"),
    httpx.RemoteProtocolError("Server disconnected without sending a response."),
    httpx.WriteError("write failed"),
])
async def test_transport_errors_before_output_are_retried(error: Exception) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise error
        return httpx.Response(200, text=_CHAT_OK)

    events = [e async for e in _chat_provider(handler).stream_response(
        model="gpt-4o", system="s", messages=[HumanEntry(content="q")], tools=[],
    )]
    assert isinstance(events[-1], StreamCloseEvent)
    assert events[-1].message.text == "hi" and len(calls) == 2


async def test_tls_error_after_output_ends_the_reply_without_a_retry() -> None:
    calls: list[int] = []

    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"index":0,"delta":{"content":"half"}}]}\n\n'
            raise ssl.SSLError(1, "bad record mac")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, stream=Broken())

    events = [e async for e in _chat_provider(handler).stream_response(
        model="gpt-4o", system="s", messages=[HumanEntry(content="q")], tools=[],
    )]
    reply = events[-1].error  # type: ignore[union-attr]
    assert reply.stop_reason == "error" and "bad record mac" in (reply.error_message or "")
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Context size after compaction
# ---------------------------------------------------------------------------


def test_usage_from_before_a_compaction_is_not_the_anchor() -> None:
    from delta_app.context.budget import estimate_transcript_tokens
    from delta_harness.contracts.transcript import PruneSummaryEntry, UsageStats

    old = ModelEntry(content=[TextSegment(text="kept")], stop_reason="stop",
                     usage=UsageStats(input=50_000), timestamp=1_000)
    summary = PruneSummaryEntry(summary="short", tokens_before=0, timestamp=2_000)
    assert estimate_transcript_tokens([summary, old]) < 1_000
    fresh = ModelEntry(content=[TextSegment(text="new")], stop_reason="stop",
                       usage=UsageStats(input=700), timestamp=3_000)
    assert estimate_transcript_tokens([summary, old, fresh]) == 700


def test_failed_reply_usage_is_not_the_anchor() -> None:
    from delta_app.context.budget import estimate_transcript_tokens
    from delta_harness.contracts.transcript import UsageStats

    failed = ModelEntry(stop_reason="error", error_message="x", usage=UsageStats(input=90_000))
    assert estimate_transcript_tokens([HumanEntry(content="q"), failed]) < 1_000


def test_replayed_compaction_keeps_its_own_time() -> None:
    from delta_harness.session.records import PruneRecord, TranscriptRecord
    from delta_harness.session.replay import project_records

    first = TranscriptRecord(message=HumanEntry(content="a"))
    prune = PruneRecord(summary="s", replaces_entry_ids=[first.id], timestamp=1_700_000_020.0)
    state = project_records([first, prune])
    assert state.transcript[0].timestamp == 1_700_000_020_000


# ---------------------------------------------------------------------------
# No naming request after a failed run
# ---------------------------------------------------------------------------


async def test_failed_run_does_not_ask_the_provider_for_a_title(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="busy")

    session = await CodingSession.create(
        provider=_chat_provider(handler), provider_name="openai", model="gpt-4o",
        system="s", sessions_dir=tmp_path,
    )
    _ = [e async for e in session.submit("hello")]
    assert len(calls) == 3
    assert session.title == "hello"


# ---------------------------------------------------------------------------
# Streams that close before the response ends
# ---------------------------------------------------------------------------

_TRUNC = "The response stream ended before the reply was complete."


def _anthropic_provider(handler):  # type: ignore[no-untyped-def]
    from delta_model.claude import AnthropicProvider
    from delta_model.settings import AnthropicProfile

    provider = AnthropicProvider(AnthropicProfile(
        name="anthropic",
        credential=Credential(api_key="k", base_url="https://api.anthropic.com"),
        retry=RetryPolicy(max_retries=1, max_delay_seconds=0.01),
    ))
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def _a(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


_A_START = _a("message_start", {"type": "message_start",
                                "message": {"id": "m", "model": "c", "usage": {}}})
_A_TEXT = (
    _a("content_block_start", {"type": "content_block_start", "index": 0,
                               "content_block": {"type": "text", "text": ""}})
    + _a("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "half"}})
)
_A_END = (
    _a("content_block_stop", {"type": "content_block_stop", "index": 0})
    + _a("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 1}})
    + _a("message_stop", {"type": "message_stop"})
)


async def _last(provider, model: str):  # type: ignore[no-untyped-def]
    events = [e async for e in provider.stream_response(
        model=model, system="s", messages=[HumanEntry(content="q")], tools=[],
    )]
    return events[-1]


async def test_anthropic_empty_close_is_retried() -> None:
    bodies = ["", _A_START + _A_TEXT + _A_END]
    last = await _last(_anthropic_provider(lambda r: httpx.Response(200, text=bodies.pop(0))),
                       "claude")
    assert isinstance(last, StreamCloseEvent) and last.message.text == "half"


async def test_anthropic_cut_mid_reply_fails_and_keeps_the_text() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, text=_A_START + _A_TEXT)

    last = await _last(_anthropic_provider(handler), "claude")
    assert last.error.stop_reason == "error" and last.error.error_message == _TRUNC
    assert last.error.text == "half" and len(calls) == 1


async def test_openai_cut_mid_reply_fails_and_keeps_the_text() -> None:
    body = 'data: {"choices":[{"index":0,"delta":{"content":"half"}}]}\n\n'
    last = await _last(_chat_provider(lambda r: httpx.Response(200, text=body)), "gpt-4o")
    assert last.error.error_message == _TRUNC and last.error.text == "half"


async def test_openai_empty_close_is_retried() -> None:
    bodies = ["", _CHAT_OK]
    last = await _last(_chat_provider(lambda r: httpx.Response(200, text=bodies.pop(0))),
                       "gpt-4o")
    assert isinstance(last, StreamCloseEvent) and last.message.text == "hi"


async def test_finish_reason_without_done_still_completes() -> None:
    body = (
        'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    )
    last = await _last(_chat_provider(lambda r: httpx.Response(200, text=body)), "gpt-4o")
    assert isinstance(last, StreamCloseEvent) and last.message.text == "ok"


async def test_responses_incomplete_is_a_length_stop_not_a_cut() -> None:
    reply = await _responses(_frames(
        {"type": "response.output_text.delta", "delta": "long"},
        {"type": "response.incomplete",
         "response": {"status": "incomplete",
                      "incomplete_details": {"reason": "max_output_tokens"}}},
        named=True,
    ))
    assert reply.stop_reason == "length" and reply.text == "long"
