"""Tests for CodingSession — the persistent, resumable coding session runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from delta_app.conversation import CodingSession, SessionStats
from delta_harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    MessageStartEvent,
    RunEndEvent,
    RunStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from delta_harness.contracts.tooling import ToolSpec
from delta_harness.contracts.transcript import (
    HumanEntry,
    ModelEntry,
    PruneSummaryEntry,
    TextSegment,
)
from delta_harness.contracts.transcript.diagnostics import CostBreakdown, UsageStats
from delta_harness.provider.wire import StreamCloseEvent
from delta_harness.session.records import (
    ModelSwapRecord,
    PruneRecord,
    ReasoningLevelRecord,
    SessionMetaRecord,
    SessionRecord,
    TagRecord,
    TipRecord,
    TranscriptRecord,
)
from delta_harness.session.replay import find_tip, project_records
from delta_harness.session.store import JsonlVault
from delta_model.scripted import ReplayProvider

# ── helpers ────────────────────────────────────────────────────────────────


def _make_reply(
    text: str = "Hello!",
    model: str = "test-model",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost: float = 0.001,
) -> ModelEntry:
    """Build a ModelEntry with usage stats for testing."""
    return ModelEntry(
        model=model,
        content=[TextSegment(text=text)],
        stop_reason="stop",
        usage=UsageStats(
            total_tokens=input_tokens + output_tokens,
            input=input_tokens,
            output=output_tokens,
            cost=CostBreakdown(total=cost),
        ),
    )


def _make_stream(reply: ModelEntry | None = None) -> list[list[StreamCloseEvent]]:
    """Build a single-reply stream for ReplayProvider."""
    if reply is None:
        reply = _make_reply()
    return [[StreamCloseEvent(reason="stop", message=reply)]]


def _make_provider(
    replies: list[ModelEntry] | None = None,
) -> ReplayProvider:
    """Build a ReplayProvider with one stream per reply."""
    if replies is None:
        replies = [_make_reply()]
    streams = [
        [StreamCloseEvent(reason="stop", message=r)] for r in replies
    ]
    return ReplayProvider(streams)


async def _collect_events(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    """Exhaust an event stream and return all events."""
    events: list[AgentEvent] = []
    async for event in stream:
        events.append(event)
    return events


# ── test: construction ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_ephemeral_session() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    assert session.session_id
    assert session.model == "test-model"
    assert session.provider_name == "test"
    assert session.transcript == ()
    assert session.title is None
    assert not session.active
    await session.shutdown()


@pytest.mark.asyncio
async def test_create_persistent_session(tmp_path: Path) -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )

    # Vault file was created with a SessionMetaRecord
    vault = JsonlVault(tmp_path / f"{session.session_id}.jsonl")
    records = await vault.read_all()
    assert any(isinstance(r, SessionMetaRecord) for r in records)
    await session.shutdown()


# ── test: submit and events ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_streams_events() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    events = await _collect_events(session.submit("Hi"))

    # Should contain run lifecycle events and message events
    types = {type(e) for e in events}
    assert RunStartEvent in types
    assert RunEndEvent in types
    assert MessageStartEvent in types
    assert MessageEndEvent in types
    assert TurnStartEvent in types
    assert TurnEndEvent in types
    await session.shutdown()


@pytest.mark.asyncio
async def test_submit_updates_transcript() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    await _collect_events(session.submit("Hi"))

    transcript = session.transcript
    assert len(transcript) == 2  # user + assistant
    assert isinstance(transcript[0], HumanEntry)
    assert isinstance(transcript[1], ModelEntry)
    assert transcript[0].text == "Hi"
    assert transcript[1].text == "Hello!"
    await session.shutdown()


@pytest.mark.asyncio
async def test_submit_updates_stats() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    await _collect_events(session.submit("Hi"))

    stats = session.usage
    assert stats.turn_count == 1
    assert stats.message_count == 2
    assert stats.total_input_tokens == 100
    assert stats.total_output_tokens == 50
    assert stats.total_cost_usd == pytest.approx(0.001)
    await session.shutdown()


@pytest.mark.asyncio
async def test_submit_persists_entries(tmp_path: Path) -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )

    await _collect_events(session.submit("Hi"))
    await session.shutdown()

    # Check vault has transcript records + tip
    vault = JsonlVault(tmp_path / f"{session.session_id}.jsonl")
    records = await vault.read_all()
    transcript_records = [r for r in records if isinstance(r, TranscriptRecord)]
    tip_records = [r for r in records if isinstance(r, TipRecord)]
    assert len(transcript_records) == 2  # user + assistant
    assert len(tip_records) >= 1


# ── test: resume ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_restores_transcript(tmp_path: Path) -> None:
    # Create and populate a session
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )
    await _collect_events(session.submit("Hi"))
    sid = session.session_id
    await session.shutdown()

    # Resume it
    provider2 = _make_provider()
    resumed = await CodingSession.resume(
        sid,
        provider=provider2,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )

    assert len(resumed.transcript) == 2
    assert isinstance(resumed.transcript[0], HumanEntry)
    assert isinstance(resumed.transcript[1], ModelEntry)
    assert resumed.usage.turn_count == 1
    await resumed.shutdown()


@pytest.mark.asyncio
async def test_resume_nonexistent_raises(tmp_path: Path) -> None:
    provider = _make_provider()
    with pytest.raises(FileNotFoundError):
        await CodingSession.resume(
            "nonexistent",
            provider=provider,
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            sessions_dir=tmp_path,
        )


# ── test: model switching ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_switch_model(tmp_path: Path) -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="model-a",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )

    await session.switch_model("model-b")

    assert session.model == "model-b"
    assert session.harness.settings.model == "model-b"

    # Check persistence
    vault = JsonlVault(tmp_path / f"{session.session_id}.jsonl")
    records = await vault.read_all()
    swaps = [r for r in records if isinstance(r, ModelSwapRecord)]
    assert len(swaps) == 1
    assert swaps[0].model == "model-b"
    await session.shutdown()


# ── test: thinking mode ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_thinking(tmp_path: Path) -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )

    await session.set_thinking("high")
    assert session.thinking_level == "high"

    await session.set_thinking(None)
    assert session.thinking_level is None

    vault = JsonlVault(tmp_path / f"{session.session_id}.jsonl")
    records = await vault.read_all()
    levels = [r for r in records if isinstance(r, ReasoningLevelRecord)]
    assert len(levels) == 2
    assert levels[0].thinking_level == "high"
    assert levels[1].thinking_level is None
    await session.shutdown()


# ── test: compaction ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_too_short_raises() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    with pytest.raises(ValueError, match="need at least"):
        await session.compact()
    await session.shutdown()


@pytest.mark.asyncio
async def test_compact_replaces_entries() -> None:
    # Build a session with enough entries to compact
    replies = [_make_reply(f"Reply {i}") for i in range(6)]
    provider = _make_provider(replies)

    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    # Submit 6 times to get 12 entries (6 user + 6 assistant)
    for i in range(6):
        await _collect_events(session.submit(f"Message {i}"))

    assert len(session.transcript) == 12

    # The provider needs one more stream for the summary generation
    summary_reply = _make_reply("Conversation summary: discussed messages 0-5")
    provider._streams.append(
        [StreamCloseEvent(reason="stop", message=summary_reply)]
    )

    entry = await session.compact()

    assert isinstance(entry, PruneSummaryEntry)
    assert "summary" in entry.summary.lower() or len(entry.summary) > 0
    # Transcript should be shorter: 1 summary + keep_count entries
    assert len(session.transcript) < 12
    assert isinstance(session.transcript[0], PruneSummaryEntry)
    await session.shutdown()


@pytest.mark.asyncio
async def test_compact_persists_prune_record(tmp_path: Path) -> None:
    replies = [_make_reply(f"Reply {i}") for i in range(6)]
    provider = _make_provider(replies)

    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )

    for i in range(6):
        await _collect_events(session.submit(f"Message {i}"))

    summary_reply = _make_reply("Summary of the conversation")
    provider._streams.append(
        [StreamCloseEvent(reason="stop", message=summary_reply)]
    )
    await session.compact()

    vault = JsonlVault(tmp_path / f"{session.session_id}.jsonl")
    records = await vault.read_all()
    prune_records = [r for r in records if isinstance(r, PruneRecord)]
    assert len(prune_records) == 1
    assert len(prune_records[0].replaces_entry_ids) > 0
    await session.shutdown()


# ── test: should_compact ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_should_compact_false_when_small() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    assert not session.should_compact()
    await session.shutdown()


# ── test: slash commands ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_command_model() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    result = await session.handle_command("/model")
    assert result is not None
    assert "test-model" in result

    result = await session.handle_command("/model new-model")
    assert result is not None
    assert "new-model" in result
    assert session.model == "new-model"
    await session.shutdown()


@pytest.mark.asyncio
async def test_handle_command_stats() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    await _collect_events(session.submit("Hi"))

    result = await session.handle_command("/stats")
    assert result is not None
    assert "Turns: 1" in result
    await session.shutdown()


@pytest.mark.asyncio
async def test_handle_command_think() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    result = await session.handle_command("/think high")
    assert result is not None
    assert session.thinking_level == "high"

    result = await session.handle_command("/think off")
    assert result is not None
    assert session.thinking_level is None
    await session.shutdown()


@pytest.mark.asyncio
async def test_handle_command_name(tmp_path: Path) -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )

    result = await session.handle_command("/name My Session")
    assert result is not None
    assert session.title == "My Session"

    vault = JsonlVault(tmp_path / f"{session.session_id}.jsonl")
    records = await vault.read_all()
    tags = [r for r in records if isinstance(r, TagRecord)]
    assert len(tags) == 1
    assert tags[0].label == "My Session"
    await session.shutdown()


@pytest.mark.asyncio
async def test_handle_command_diag() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    result = await session.handle_command("/diag")
    assert result is not None
    import json
    diag = json.loads(result)
    assert diag["model"] == "test-model"
    assert diag["provider"] == "test"
    await session.shutdown()


@pytest.mark.asyncio
async def test_handle_command_unknown_returns_none() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    result = await session.handle_command("/nonexistent")
    assert result is None
    await session.shutdown()


@pytest.mark.asyncio
async def test_handle_command_not_slash_returns_none() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    result = await session.handle_command("hello")
    assert result is None
    await session.shutdown()


# ── test: shell execution ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_shell() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    entry = await session.run_shell("echo hello")
    assert entry.exit_code == 0
    assert "hello" in entry.output
    # Should NOT be in transcript
    assert len(session.transcript) == 0
    await session.shutdown()


@pytest.mark.asyncio
async def test_run_shell_inject() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    entry = await session.run_shell("echo injected", inject=True)
    assert entry.exit_code == 0
    # Should be in transcript
    assert len(session.transcript) == 1
    assert session.transcript[0].role == "bashExecution"
    await session.shutdown()


# ── test: branching ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_branch_with_summary(tmp_path: Path) -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )

    branch_id = await session.branch(summary="Test branch")
    assert branch_id

    vault = JsonlVault(tmp_path / f"{session.session_id}.jsonl")
    records = await vault.read_all()
    from delta_harness.session.records import ForkSummaryRecord

    forks = [r for r in records if isinstance(r, ForkSummaryRecord)]
    assert len(forks) == 1
    assert forks[0].summary == "Test branch"
    await session.shutdown()


# ── test: rewind ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rewind(tmp_path: Path) -> None:
    # Create session, submit twice, rewind to first exchange
    reply1 = _make_reply("First reply")
    reply2 = _make_reply("Second reply")
    provider = _make_provider([reply1, reply2])
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )

    await _collect_events(session.submit("First"))
    # Record the ID of the first assistant message
    first_records = list(session._record_ids)
    assert len(first_records) >= 2
    rewind_target = first_records[1]  # the first ModelEntry record

    await _collect_events(session.submit("Second"))
    assert len(session.transcript) == 4  # 2 user + 2 assistant

    await session.rewind(rewind_target)

    # Should have rewound to first exchange
    assert len(session.transcript) == 2
    assert isinstance(session.transcript[1], ModelEntry)
    assert session.transcript[1].text == "First reply"
    await session.shutdown()


@pytest.mark.asyncio
async def test_rewind_requires_persistent() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    with pytest.raises(RuntimeError, match="persistent"):
        await session.rewind("some-id")
    await session.shutdown()


# ── test: multi-session lifecycle ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_new_session(tmp_path: Path) -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )
    old_id = session.session_id

    new = await session.new_session()

    assert new.session_id != old_id
    assert new.model == "test-model"
    assert new.provider_name == "test"
    assert new.transcript == ()
    await new.shutdown()


@pytest.mark.asyncio
async def test_replace_session(tmp_path: Path) -> None:
    # Create two sessions
    provider1 = _make_provider()
    s1 = await CodingSession.create(
        provider=provider1,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )
    await _collect_events(s1.submit("Hi from s1"))
    s1_id = s1.session_id

    provider2 = _make_provider()
    s2 = await CodingSession.create(
        provider=provider2,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )
    _s2_id = s2.session_id
    await s2.shutdown()

    # Replace s2 with s1
    replaced = await s2.replace_session(s1_id)
    assert replaced.session_id == s1_id
    assert len(replaced.transcript) == 2
    await replaced.shutdown()


@pytest.mark.asyncio
async def test_replace_requires_persistent() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    with pytest.raises(RuntimeError, match="persistent"):
        await session.replace_session("some-id")
    await session.shutdown()


# ── test: export ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_export_text() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    await _collect_events(session.submit("Hi"))

    text = await session.export("text")
    assert "[user]" in text
    assert "[assistant]" in text
    await session.shutdown()


@pytest.mark.asyncio
async def test_export_json() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    await _collect_events(session.submit("Hi"))

    import json
    text = await session.export("json")
    data = json.loads(text)
    assert isinstance(data, list)
    assert len(data) == 2
    await session.shutdown()


@pytest.mark.asyncio
async def test_export_jsonl_persistent(tmp_path: Path) -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )

    await _collect_events(session.submit("Hi"))

    text = await session.export("jsonl")
    lines = [level for level in text.strip().split("\n") if level.strip()]
    assert len(lines) >= 3  # meta + user + assistant (+ tip)
    await session.shutdown()


# ── test: diagnostics ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_diagnostics() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    diag = session.diagnostics()

    assert diag["session_id"] == session.session_id
    assert diag["model"] == "test-model"
    assert diag["provider"] == "test"
    assert diag["active"] is False
    assert diag["persistent"] is False
    await session.shutdown()


# ── test: queues ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inject_and_enqueue() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    session.inject("steering message")
    session.enqueue("follow up")

    pending = session.pending
    assert pending.count == 2

    flushed = session.flush_queues()
    assert flushed.count == 2
    assert session.pending.count == 0
    await session.shutdown()


# ── test: abort ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_abort_no_crash() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    # Aborting when not running should not crash
    session.abort()
    await session.shutdown()


# ── test: read-only API ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_only_properties() -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        cwd="/test/dir",
    )

    assert session.session_id
    assert session.model == "test-model"
    assert session.provider_name == "test"
    assert session.tools == ()
    assert not session.active
    assert session.title is None
    assert session.cwd == "/test/dir"
    assert session.thinking_level is None
    assert isinstance(session.usage, SessionStats)
    assert session.harness is not None
    await session.shutdown()


# ── test: tools_loader ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reload_with_tools_loader() -> None:
    calls: list[int] = []

    def loader() -> list[ToolSpec]:
        calls.append(1)
        return []

    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
        tools_loader=loader,
    )

    await session.reload()
    assert len(calls) == 1
    await session.shutdown()


# ── test: projection ──────────────────────────────────────────────────────


def test_project_records_empty() -> None:
    state = project_records([])
    assert state.transcript == []
    assert state.record_ids == []
    assert state.model is None
    assert state.title is None


def test_project_records_transcript() -> None:
    user = HumanEntry(content="Hi")
    reply = _make_reply()
    records: list[SessionRecord] = [
        SessionMetaRecord(id="meta", cwd="/test"),
        TranscriptRecord(id="r1", message=user),
        TranscriptRecord(id="r2", message=reply),
    ]

    state = project_records(records)
    assert len(state.transcript) == 2
    assert state.record_ids == ["r1", "r2"]
    assert state.cwd == "/test"


def test_project_records_with_prune() -> None:
    user = HumanEntry(content="Hi")
    reply = _make_reply()
    records: list[SessionRecord] = [
        TranscriptRecord(id="r1", message=user),
        TranscriptRecord(id="r2", message=reply),
        TranscriptRecord(id="r3", message=HumanEntry(content="Follow-up")),
        PruneRecord(
            id="prune1",
            summary="Discussed greetings",
            replaces_entry_ids=["r1", "r2"],
        ),
        TranscriptRecord(id="r4", message=_make_reply("OK")),
    ]

    state = project_records(records)
    # r1 and r2 are superseded; the summary takes the position of the first
    # record it replaces (r1), so it leads rather than trailing the survivors.
    assert len(state.transcript) == 3
    assert isinstance(state.transcript[0], PruneSummaryEntry)  # prune1, at r1
    assert state.transcript[0].summary == "Discussed greetings"
    assert isinstance(state.transcript[1], HumanEntry)  # r3
    assert state.record_ids == ["prune1", "r3", "r4"]


def test_project_records_prune_supersedes_earlier_summary() -> None:
    """A later compaction replaces an earlier summary, not just messages."""
    records: list[SessionRecord] = [
        TranscriptRecord(id="r1", message=HumanEntry(content="Hi")),
        PruneRecord(id="prune1", summary="First pass", replaces_entry_ids=["r1"]),
        TranscriptRecord(id="r2", message=HumanEntry(content="More")),
        PruneRecord(
            id="prune2",
            summary="Second pass",
            replaces_entry_ids=["prune1", "r2"],
        ),
        TranscriptRecord(id="r3", message=HumanEntry(content="Latest")),
    ]

    state = project_records(records)
    assert state.record_ids == ["prune2", "r3"]
    assert isinstance(state.transcript[0], PruneSummaryEntry)
    assert state.transcript[0].summary == "Second pass"


def test_project_records_model_and_thinking() -> None:
    records: list[SessionRecord] = [
        ModelSwapRecord(id="m1", model="gpt-4o"),
        ReasoningLevelRecord(id="t1", thinking_level="high"),
        TagRecord(id="tag1", label="My Session"),
    ]

    state = project_records(records)
    assert state.model == "gpt-4o"
    assert state.thinking_level == "high"
    assert state.title == "My Session"


def test_find_tip_empty() -> None:
    assert find_tip([]) is None


def test_find_tip() -> None:
    records: list[SessionRecord] = [
        TipRecord(id="t1", entry_id="abc"),
        TipRecord(id="t2", entry_id="def"),
    ]
    assert find_tip(records) == "def"


# ── test: multiple submits accumulate stats ────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_submits_accumulate() -> None:
    replies = [
        _make_reply("R1", input_tokens=100, output_tokens=50, cost=0.001),
        _make_reply("Auto Title", input_tokens=10, output_tokens=5, cost=0.0),
        _make_reply("R2", input_tokens=200, output_tokens=80, cost=0.002),
    ]
    provider = _make_provider(replies)
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    # First submit triggers auto-naming (consumes an extra stream)
    await _collect_events(session.submit("First"))
    await _collect_events(session.submit("Second"))

    stats = session.usage
    assert stats.turn_count == 2
    assert stats.message_count == 4
    assert stats.total_input_tokens == 300
    assert stats.total_output_tokens == 130
    assert stats.total_cost_usd == pytest.approx(0.003)
    await session.shutdown()


# ── test: resume preserves model switch ────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_preserves_model_switch(tmp_path: Path) -> None:
    provider = _make_provider()
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="model-a",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )
    await session.switch_model("model-b")
    sid = session.session_id
    await session.shutdown()

    provider2 = _make_provider()
    resumed = await CodingSession.resume(
        sid,
        provider=provider2,
        provider_name="test",
        model="model-a",
        system="You are helpful.",
        sessions_dir=tmp_path,
    )
    assert resumed.model == "model-b"
    await resumed.shutdown()


# ── test: context estimation ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_estimation() -> None:
    provider = _make_provider([_make_reply(input_tokens=5000)])
    session = await CodingSession.create(
        provider=provider,
        provider_name="test",
        model="test-model",
        system="You are helpful.",
    )

    await _collect_events(session.submit("Hi"))

    stats = session.usage
    # Estimated context should be at least the input tokens from the reply
    assert stats.estimated_context_tokens >= 5000
    await session.shutdown()


# ── wiring regressions ─────────────────────────────────────────────────────
#
# Both capabilities below existed as complete modules but were unreachable at
# run time. These tests fail if the wiring is ever removed again.


class TestThinkingLevelReachesProvider:
    class _RecordingProvider:
        def __init__(self) -> None:
            from delta_model.settings import ReasoningPolicy

            self.policy = ReasoningPolicy()

        def set_reasoning(self, policy) -> None:  # noqa: ANN001
            self.policy = policy

        def stream_response(self, **kwargs):  # noqa: ANN003, ANN201
            raise NotImplementedError

    async def _session(self, tmp_path: Path):  # noqa: ANN202
        from delta_app.conversation import CodingSession

        provider = self._RecordingProvider()
        session = await CodingSession.create(
            provider=provider,
            provider_name="rec",
            model="m",
            system="s",
            tools=[],
            sessions_dir=tmp_path,
        )
        return session, provider

    async def test_set_thinking_updates_provider_policy(self, tmp_path: Path) -> None:
        session, provider = await self._session(tmp_path)
        assert provider.policy.enabled is False

        await session.set_thinking("high")
        assert provider.policy.enabled is True
        high_budget = provider.policy.budget_tokens

        await session.set_thinking("low")
        assert provider.policy.budget_tokens is not None
        assert provider.policy.budget_tokens < high_budget

    async def test_thinking_off_disables_policy(self, tmp_path: Path) -> None:
        session, provider = await self._session(tmp_path)
        await session.set_thinking("high")
        await session.set_thinking(None)
        assert provider.policy.enabled is False

    async def test_provider_without_set_reasoning_is_tolerated(
        self, tmp_path: Path,
    ) -> None:
        from delta_app.conversation import CodingSession

        class Bare:
            def stream_response(self, **kwargs):  # noqa: ANN003, ANN201
                raise NotImplementedError

        session = await CodingSession.create(
            provider=Bare(),
            provider_name="bare",
            model="m",
            system="s",
            tools=[],
            sessions_dir=tmp_path,
        )
        await session.set_thinking("high")  # must not raise
        assert session.thinking_level == "high"


class TestPromptTemplatesAreLoaded:
    async def test_templates_load_and_expand(self, tmp_path: Path) -> None:
        from delta_app.cli.main import _load_prompt_templates
        from delta_app.conversation import CodingSession

        prompts_dir = tmp_path / ".delta" / "prompts"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "review.md").write_text(
            "---\nname: review\ndescription: Review code\n---\n"
            "Please review {{arguments}} carefully.\n",
            encoding="utf-8",
        )

        session = await CodingSession.create(
            provider=_make_provider(),
            provider_name="stub",
            model="m",
            system="s",
            tools=[],
            sessions_dir=tmp_path / "sessions",
            templates_loader=lambda: _load_prompt_templates(str(tmp_path)),
        )

        assert [t.name for t in session.prompt_templates] == ["review"]
        expanded = session.expand_template("/review the auth module")
        assert expanded is not None
        assert "the auth module" in expanded

    async def test_non_template_command_not_swallowed(self, tmp_path: Path) -> None:
        from delta_app.conversation import CodingSession

        session = await CodingSession.create(
            provider=_make_provider(),
            provider_name="stub",
            model="m",
            system="s",
            tools=[],
            sessions_dir=tmp_path,
        )
        assert session.expand_template("/stats") is None
