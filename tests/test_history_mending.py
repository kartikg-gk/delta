"""Repair of malformed tool-call history: pure policy, loop boundary, and disk."""

from __future__ import annotations

from pathlib import Path

from delta_app.conversation import CodingSession
from delta_harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ModelEntry,
    TextSegment,
    ToolOutcomeEntry,
)
from delta_harness.driver import RuntimeConfig, RuntimeHarness
from delta_harness.mending import INTERRUPTED_RESULT_TEXT, mend_tool_history
from delta_harness.provider.wire import StreamCloseEvent
from delta_harness.session.records import ExtensionRecord, TipRecord, TranscriptRecord
from delta_harness.session.replay import find_tip
from delta_harness.session.store import JsonlVault
from delta_model.scripted import ReplayProvider


def _calls(*ids: str) -> ModelEntry:
    return ModelEntry(content=[CallBlock(id=i, name="read", arguments={}) for i in ids])


def _result(call_id: str, text: str = "ok", *, error: bool = False) -> ToolOutcomeEntry:
    return ToolOutcomeEntry(tool_call_id=call_id, tool_name="read", content=text, is_error=error)


def _shape(entries) -> list[str]:
    out = []
    for e in entries:
        if isinstance(e, ToolOutcomeEntry):
            out.append(f"r:{e.tool_call_id}:{e.text}")
        elif isinstance(e, ModelEntry):
            out.append("a:" + ",".join(c.id for c in e.tool_calls))
        else:
            out.append("u")
    return out


# ---------------------------------------------------------------------------
# Pure policy
# ---------------------------------------------------------------------------


def test_valid_history_is_untouched() -> None:
    history = [HumanEntry(content="q"), _calls("a"), _result("a")]
    mend = mend_tool_history(history)
    assert not mend.changed
    assert all(x is y for x, y in zip(mend.entries, history, strict=True))


def test_missing_result_is_synthesized() -> None:
    mend = mend_tool_history([HumanEntry(content="q"), _calls("a"), HumanEntry(content="next")])
    assert _shape(mend.entries) == ["u", "a:a", f"r:a:{INTERRUPTED_RESULT_TEXT}", "u"]
    assert mend.entries[2].is_error
    assert mend.synthesized == 1


def test_orphan_result_is_dropped() -> None:
    mend = mend_tool_history([HumanEntry(content="q"), _result("ghost"), _calls("a"), _result("a")])
    assert _shape(mend.entries) == ["u", "a:a", "r:a:ok"]
    assert mend.dropped_orphans == 1


def test_separated_result_moves_next_to_its_call() -> None:
    mend = mend_tool_history([_calls("a"), HumanEntry(content="x"), _result("a")])
    assert _shape(mend.entries) == ["a:a", "r:a:ok", "u"]
    assert mend.reordered == 1


def test_parallel_results_follow_call_order() -> None:
    mend = mend_tool_history([_calls("a", "b"), _result("b", "B"), _result("a", "A")])
    assert _shape(mend.entries) == ["a:a,b", "r:a:A", "r:b:B"]


def test_duplicate_results_collapse_real_one_wins() -> None:
    history = [
        _calls("a"),
        _result("a", INTERRUPTED_RESULT_TEXT, error=True),
        _result("a", "real"),
    ]
    mend = mend_tool_history(history)
    assert _shape(mend.entries) == ["a:a", "r:a:real"]
    assert mend.dropped_duplicates == 1


def test_mending_twice_is_a_no_op() -> None:
    first = mend_tool_history([_calls("a", "b"), _result("b"), HumanEntry(content="x")])
    second = mend_tool_history(first.entries)
    assert not second.changed


# ---------------------------------------------------------------------------
# Loop boundary: provider sees mended input, harness history is untouched
# ---------------------------------------------------------------------------


class _Recorder:
    """Provider that records each request's keyword arguments."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def stream_response(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        reply = ModelEntry(content=[TextSegment(text="fine")], stop_reason="stop")
        yield StreamCloseEvent(reason="stop", message=reply)


async def test_provider_input_is_mended_without_touching_stored_history() -> None:
    provider = _Recorder()
    broken = [HumanEntry(content="q"), _result("ghost")]
    harness = RuntimeHarness(
        RuntimeConfig(provider=provider, model="m", system="s", cache_key="sess-9"),
        messages=list(broken),
    )
    _ = [e async for e in harness.submit("go")]

    sent = provider.requests[0]
    assert not any(isinstance(e, ToolOutcomeEntry) for e in sent["messages"])
    assert sent["cache_key"] == "sess-9"
    assert any(isinstance(e, ToolOutcomeEntry) for e in harness.transcript)


# ---------------------------------------------------------------------------
# Durable repair on resume
# ---------------------------------------------------------------------------


async def _corrupt_session(tmp_path: Path) -> str:
    ok = ModelEntry(content=[TextSegment(text="hi")], stop_reason="stop")
    provider = ReplayProvider([[StreamCloseEvent(reason="stop", message=ok)]])
    session = await CodingSession.create(
        provider=provider, provider_name="test", model="m", system="s", sessions_dir=tmp_path,
    )
    _ = [e async for e in session.submit("first")]
    sid = session.session_id
    await session.shutdown()

    vault = JsonlVault(tmp_path / f"{sid}.jsonl")
    records = await vault.read_all()
    parent = find_tip(records)
    for entry in (_calls("a", "b"), _result("b"), _result("ghost")):
        record = TranscriptRecord(parent_id=parent, message=entry)
        await vault.append(record)
        parent = record.id
    await vault.append(TipRecord(entry_id=parent))
    return sid


async def _resume(sid: str, tmp_path: Path) -> CodingSession:
    return await CodingSession.resume(
        sid, provider=ReplayProvider([]), provider_name="test", model="m", system="s",
        sessions_dir=tmp_path,
    )


async def test_resume_appends_one_repair_branch(tmp_path: Path) -> None:
    sid = await _corrupt_session(tmp_path)
    vault = JsonlVault(tmp_path / f"{sid}.jsonl")
    before = await vault.read_all()

    session = await _resume(sid, tmp_path)
    assert _shape(session.transcript)[-3:] == [
        "a:a,b", f"r:a:{INTERRUPTED_RESULT_TEXT}", "r:b:ok",
    ]

    after = await vault.read_all()
    assert after[: len(before)] == before  # nothing rewritten
    repairs = [r for r in after if isinstance(r, ExtensionRecord)
               and r.namespace == "delta.history-repair"]
    assert len(repairs) == 1
    assert repairs[0].data["synthesized"] == 1
    assert repairs[0].data["dropped_orphans"] == 1
    await session.shutdown()


async def test_second_resume_adds_no_more_repairs(tmp_path: Path) -> None:
    sid = await _corrupt_session(tmp_path)
    await (await _resume(sid, tmp_path)).shutdown()
    session = await _resume(sid, tmp_path)

    records = await JsonlVault(tmp_path / f"{sid}.jsonl").read_all()
    repairs = [r for r in records if isinstance(r, ExtensionRecord)
               and r.namespace == "delta.history-repair"]
    assert len(repairs) == 1
    assert _shape(session.transcript)[-2:] == [f"r:a:{INTERRUPTED_RESULT_TEXT}", "r:b:ok"]
    await session.shutdown()
