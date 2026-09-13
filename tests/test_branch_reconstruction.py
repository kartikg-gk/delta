"""Active-branch reconstruction on resume, and /continue after a stop."""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, str(__file__.rsplit("test_branch_reconstruction.py", 1)[0]))

from _session_helpers import _make_session, _text_turn  # noqa: E402

from delta_app.conversation import CodingSession  # noqa: E402
from delta_harness.contracts.transcript import surface_text  # noqa: E402
from delta_harness.session.records import TipRecord, TranscriptRecord  # noqa: E402
from delta_harness.session.replay import (  # noqa: E402
    TreeIntegrityError,
    find_tip,
    project_active_branch,
)
from delta_model.scripted import ReplayProvider  # noqa: E402


def _texts(entries) -> list[str]:
    return [surface_text(e) for e in entries]


async def _resume(tmp_path, session_id) -> CodingSession:
    return await CodingSession.resume(
        session_id,
        provider=ReplayProvider([]),
        provider_name="replay",
        model="m",
        system="s",
        sessions_dir=tmp_path / "sessions",
    )


class TestActiveBranchProjection:
    def test_no_tip_falls_back_to_flat(self):
        records = [TranscriptRecord(message=__import__(
            "delta_harness.contracts.transcript", fromlist=["HumanEntry"],
        ).HumanEntry(content="hi"))]
        assert find_tip(records) is None
        state = project_active_branch(records)
        assert len(state.transcript) == 1

    def test_broken_chain_degrades_instead_of_raising(self):
        from delta_harness.contracts.transcript import HumanEntry

        orphan = TranscriptRecord(message=HumanEntry(content="hi"))
        orphan.parent_id = "does-not-exist"
        records = [orphan, TipRecord(entry_id=orphan.id)]

        with pytest.raises(TreeIntegrityError):
            from delta_harness.session.replay import trace_to_entry

            trace_to_entry(records, orphan.id)

        # The loader path must still return something usable.
        state = project_active_branch(records)
        assert len(state.transcript) == 1


class TestResumeAfterRewind:
    async def test_abandoned_branch_is_not_resurrected(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["A"])] * 12)
        sid = session.session_id

        async for _ in session.submit("MSG-ONE"):
            pass
        fork_point = session._record_ids[-1]
        async for _ in session.submit("MSG-TWO"):
            pass
        await session.rewind(fork_point)
        async for _ in session.submit("MSG-THREE"):
            pass
        live = _texts(session.transcript)
        await session.shutdown()

        resumed = await _resume(tmp_path, sid)
        loaded = _texts(resumed.transcript)
        await resumed.shutdown()

        assert loaded == live, "resume diverged from live state"
        assert not any("MSG-TWO" in t for t in loaded), "abandoned branch leaked"
        assert any("MSG-THREE" in t for t in loaded)

    async def test_plain_session_round_trips(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["A"])] * 8)
        sid = session.session_id
        async for _ in session.submit("hello"):
            pass
        live = _texts(session.transcript)
        await session.shutdown()

        resumed = await _resume(tmp_path, sid)
        loaded = _texts(resumed.transcript)
        await resumed.shutdown()
        assert loaded == live

    async def test_metadata_survives_branch_projection(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["A"])] * 8)
        sid = session.session_id
        async for _ in session.submit("hello"):
            pass
        await session.handle_command("/name Kept Title")
        await session.shutdown()

        resumed = await _resume(tmp_path, sid)
        title = resumed.title
        await resumed.shutdown()
        assert title == "Kept Title"

    async def test_tip_is_restored(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["A"])] * 8)
        sid = session.session_id
        async for _ in session.submit("hello"):
            pass
        await session.shutdown()

        resumed = await _resume(tmp_path, sid)
        tip = resumed._tip_id
        await resumed.shutdown()
        assert tip is not None, "resumed session has no tip to branch from"


class TestContinue:
    """`resume_run` was implemented but unreachable; /continue wires it up."""

    def test_continue_is_registered(self):
        assert any(name == "continue" for name, _ in CodingSession.COMMANDS)
