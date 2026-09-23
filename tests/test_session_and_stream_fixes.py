"""Regression tests: retry codes, proxy schemes, model choice on rewind and
resume, slash input in the terminal UI, and Responses stream edge cases."""

from __future__ import annotations

import json
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
