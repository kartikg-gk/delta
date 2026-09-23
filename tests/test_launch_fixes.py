"""Regressions found in the pre-launch parity sweep."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from delta_app.conversation import CodingSession
from delta_app.tools.bash import make_bash_tool
from delta_harness.contracts.stream import RunEndEvent
from delta_harness.contracts.transcript import (
    HumanEntry,
    ModelEntry,
    PruneSummaryEntry,
    TextSegment,
    ThoughtSegment,
)
from delta_harness.provider.wire import StreamCloseEvent, StreamFaultEvent
from delta_harness.session.records import TranscriptRecord
from delta_harness.session.store import JsonlVault
from delta_model._oai.normalize import EventAssembler
from delta_model._oai.parsers import ChatDecoder
from delta_model.scripted import ReplayProvider

# ---------------------------------------------------------------------------
# Session files survive Unicode line separators in message text
# ---------------------------------------------------------------------------


async def test_unicode_line_separators_round_trip(tmp_path: Path) -> None:
    vault = JsonlVault(tmp_path / "s.jsonl")
    text = "a\u2028b\u2029c\x1cd\x85e\x0bf"
    await vault.append(TranscriptRecord(message=HumanEntry(content=text)))
    await vault.append(TranscriptRecord(message=HumanEntry(content="next")))
    records = await vault.read_all()
    assert [r.message.content for r in records] == [text, "next"]


# ---------------------------------------------------------------------------
# Shell commands never read the terminal
# ---------------------------------------------------------------------------


async def test_prompting_command_fails_fast_instead_of_waiting() -> None:
    tool = make_bash_tool()
    started = time.monotonic()
    command = f'"{sys.executable}" -c "input()"'
    outcome = await tool.execute("c1", {"command": command}, None, None)
    assert time.monotonic() - started < 10
    assert "EOFError" in outcome.text


# ---------------------------------------------------------------------------
# Chat Completions reasoning and answer are independent channels
# ---------------------------------------------------------------------------


def _assemble(deltas: list[dict]) -> tuple[list, int]:
    decoder, assembler = ChatDecoder(), EventAssembler(model="m", provider="p", api="chat")
    opened = 0
    finish = {"index": 0, "delta": {}, "finish_reason": "stop"}
    frames = [{"index": 0, "delta": d} for d in deltas] + [finish]
    for frame in frames:
        for signal in decoder.decode("", {"id": "c", "choices": [frame]}):
            opened += sum(e.type.endswith("_start") for e in assembler.accept(signal))
    return assembler.finalize()[-1].message.content, opened


def test_same_chunk_reasoning_and_answer_stay_two_blocks() -> None:
    blocks, opened = _assemble(
        [{"reasoning_content": f"r{i} ", "content": f"t{i} "} for i in range(3)]
    )
    assert opened == 2
    assert isinstance(blocks[0], ThoughtSegment) and blocks[0].thinking == "r0 r1 r2 "
    assert isinstance(blocks[1], TextSegment) and blocks[1].text == "t0 t1 t2 "


def test_alternating_fragments_do_not_split_sentences() -> None:
    blocks, _ = _assemble([
        {"reasoning_content": "Done. C"}, {"content": "A"},
        {"reasoning_content": "reated."}, {"content": "B"},
    ])
    assert [getattr(b, "thinking", None) or b.text for b in blocks] == ["Done. Created.", "AB"]


def test_responses_api_keeps_sequential_blocks() -> None:
    assembler = EventAssembler(model="m", provider="p", api="responses")
    from delta_model._oai.parsers import ReasoningChunk, TextChunk

    for signal in (ReasoningChunk("a"), TextChunk("b"), ReasoningChunk("c")):
        assembler.accept(signal)
    blocks = assembler.finalize()[-1].message.content
    assert len(blocks) == 3


# ---------------------------------------------------------------------------
# Context overflow: compact, then continue the same run once
# ---------------------------------------------------------------------------

_OVERFLOW = "This model's maximum context length is 8192 tokens."


def _ok(text: str) -> list:
    return [StreamCloseEvent(reason="stop", message=ModelEntry(
        content=[TextSegment(text=text)], stop_reason="stop"))]


def _overflow() -> list:
    return [StreamFaultEvent(reason="error", error=ModelEntry(
        stop_reason="error", error_message=_OVERFLOW))]


async def _session(tmp_path: Path, streams: list) -> tuple[CodingSession, ReplayProvider]:
    provider = ReplayProvider(streams)
    session = await CodingSession.create(
        provider=provider, provider_name="test", model="m", system="s", sessions_dir=tmp_path,
    )

    async def no_title() -> str:
        return ""

    session.auto_name = no_title  # type: ignore[method-assign]
    return session, provider


async def test_overflow_compacts_and_continues_without_new_prompt(tmp_path: Path) -> None:
    streams = [_ok(f"reply {i}") for i in range(4)]
    streams += [_overflow(), _ok("summary of earlier work"), _ok("recovered")]
    session, provider = await _session(tmp_path, streams)
    for i in range(4):
        _ = [e async for e in session.submit(f"prompt {i}")]

    events = [e async for e in session.submit("big request")]

    assert sum(isinstance(e, RunEndEvent) for e in events) == 1
    transcript = session.transcript
    assert any(isinstance(e, PruneSummaryEntry) for e in transcript)
    assert transcript[-1].text == "recovered"
    assert sum(isinstance(e, HumanEntry) and e.content == "big request" for e in transcript) == 1
    assert len(provider.calls) == 7


async def test_overflow_with_nothing_to_compact_surfaces_the_error(tmp_path: Path) -> None:
    session, provider = await _session(tmp_path, [_overflow()])
    events = [e async for e in session.submit("huge")]

    assert sum(isinstance(e, RunEndEvent) for e in events) == 1
    assert session.transcript[-1].error_message == _OVERFLOW
    assert len(provider.calls) == 1


async def test_oversized_image_is_refused_not_attached(tmp_path: Path) -> None:
    from delta_app.tools.files import make_read_tool

    big = tmp_path / "shot.png"
    big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 5_100_000)
    outcome = await make_read_tool().execute("c1", {"file_path": str(big)}, None, None)
    assert "limit is 5 MB" in outcome.text
    assert not any(getattr(seg, "data", None) for seg in outcome.content)


# ---------------------------------------------------------------------------
# A session that is opened and left is never saved
# ---------------------------------------------------------------------------


async def test_unused_session_leaves_nothing_behind(tmp_path: Path) -> None:
    from delta_app.conversation import CodingSession
    from delta_harness.session.index import SessionCatalog
    from delta_model.scripted import ReplayProvider

    session = await CodingSession.create(
        provider=ReplayProvider([]), provider_name="t", model="m", system="s",
        sessions_dir=tmp_path,
    )
    fresh = await session.new_session()
    await fresh.shutdown()
    assert list(tmp_path.iterdir()) == []
    assert SessionCatalog(tmp_path).list_all() == []


async def test_first_record_saves_header_and_index_row(tmp_path: Path) -> None:
    from delta_app.conversation import CodingSession
    from delta_harness.session.index import SessionCatalog
    from delta_harness.session.records import SessionMetaRecord
    from delta_model.scripted import ReplayProvider

    session = await CodingSession.create(
        provider=ReplayProvider([]), provider_name="t", model="m", system="s",
        sessions_dir=tmp_path, cwd="/work",
    )
    await session.set_thinking("high")
    records = await session._vault.read_all()  # type: ignore[union-attr]
    assert isinstance(records[0], SessionMetaRecord) and records[0].cwd == "/work"
    assert records[1].parent_id == records[0].id
    meta = SessionCatalog(tmp_path).get(session.session_id)
    assert meta is not None and meta.cwd == "/work" and meta.provider == "t"
