"""Tests for SessionManager — application-level session lifecycle coordinator."""

from __future__ import annotations

from pathlib import Path

import pytest

from delta_app.sessions import SessionManager, SessionSummary
from delta_harness.contracts.transcript import ModelEntry, TextSegment
from delta_harness.contracts.transcript.diagnostics import CostBreakdown, UsageStats
from delta_harness.provider.wire import StreamCloseEvent
from delta_harness.session.index import SessionCatalog
from delta_model.scripted import ReplayProvider

# ── helpers ────────────────────────────────────────────────────────────────


def _make_reply(text: str = "Hello!") -> ModelEntry:
    return ModelEntry(
        model="test-model",
        content=[TextSegment(text=text)],
        stop_reason="stop",
        usage=UsageStats(
            total_tokens=150, input=100, output=50,
            cost=CostBreakdown(total=0.001),
        ),
    )


def _make_provider(replies: list[ModelEntry] | None = None) -> ReplayProvider:
    if replies is None:
        replies = [_make_reply()]
    return ReplayProvider([
        [StreamCloseEvent(reason="stop", message=r)] for r in replies
    ])


# ── construction ──────────────────────────────────────────────────────────


class TestConstruction:
    def test_sessions_dir(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        assert mgr.sessions_dir == tmp_path

    def test_catalog_property(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        assert isinstance(mgr.catalog, SessionCatalog)


# ── create ────────────────────────────────────────────────────────────────


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_returns_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        session = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            cwd="/project",
        )
        assert session.session_id
        assert session.model == "test-model"
        await session.shutdown()

    @pytest.mark.asyncio
    async def test_create_indexes_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        session = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            cwd="/project",
        )

        meta = mgr.get(session.session_id)
        assert meta is not None
        assert meta.model == "test-model"
        assert meta.provider == "test"
        assert meta.cwd == "/project"
        await session.shutdown()

    @pytest.mark.asyncio
    async def test_create_with_custom_id(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        session = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            session_id="custom-id",
        )
        assert session.session_id == "custom-id"
        assert mgr.get("custom-id") is not None
        await session.shutdown()


# ── resume ────────────────────────────────────────────────────────────────


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_restores_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        provider = _make_provider()
        session = await mgr.create(
            provider=provider,
            provider_name="test",
            model="test-model",
            system="You are helpful.",
        )
        sid = session.session_id
        await session.shutdown()

        resumed = await mgr.resume(
            sid,
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
        )
        assert resumed.session_id == sid
        await resumed.shutdown()

    @pytest.mark.asyncio
    async def test_resume_nonexistent_raises(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        with pytest.raises(FileNotFoundError):
            await mgr.resume(
                "nonexistent",
                provider=_make_provider(),
                provider_name="test",
                model="test-model",
                system="You are helpful.",
            )


# ── resume_latest ─────────────────────────────────────────────────────────


class TestResumeLatest:
    @pytest.mark.asyncio
    async def test_resume_latest_finds_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        session = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            cwd="/my/project",
        )
        sid = session.session_id
        await session.shutdown()

        latest = await mgr.resume_latest(
            "/my/project",
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
        )
        assert latest is not None
        assert latest.session_id == sid
        await latest.shutdown()

    @pytest.mark.asyncio
    async def test_resume_latest_returns_none(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        result = await mgr.resume_latest(
            "/nonexistent",
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
        )
        assert result is None


# ── list_sessions ─────────────────────────────────────────────────────────


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_empty(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        assert mgr.list_sessions() == []

    @pytest.mark.asyncio
    async def test_list_returns_summaries(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        s1 = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="model-a",
            system="You are helpful.",
            cwd="/a",
        )
        s2 = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="model-b",
            system="You are helpful.",
            cwd="/b",
        )

        summaries = mgr.list_sessions()
        assert len(summaries) == 2
        ids = {s.session_id for s in summaries}
        assert s1.session_id in ids
        assert s2.session_id in ids

        for summary in summaries:
            assert isinstance(summary, SessionSummary)
            assert summary.provider == "test"

        await s1.shutdown()
        await s2.shutdown()

    @pytest.mark.asyncio
    async def test_list_newest_first(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        s1 = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
        )
        import time
        time.sleep(0.01)
        s2 = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
        )

        summaries = mgr.list_sessions()
        assert summaries[0].session_id == s2.session_id
        assert summaries[1].session_id == s1.session_id

        await s1.shutdown()
        await s2.shutdown()


# ── fallback vault scan ───────────────────────────────────────────────────


class TestFallbackScan:
    def test_scan_vault_files_no_index(self, tmp_path: Path) -> None:
        # Write a minimal vault file without an index
        vault_path = tmp_path / "legacy123.jsonl"
        vault_path.write_text(
            '{"type":"session_info","id":"m","timestamp":1.0,"cwd":"/x"}\n'
            '{"type":"message","id":"r1","timestamp":2.0,"message":'
            '{"role":"user","content":"hi"}}\n',
            encoding="utf-8",
        )

        mgr = SessionManager(tmp_path)
        summaries = mgr.list_sessions()
        assert len(summaries) == 1
        assert summaries[0].session_id == "legacy123"
        assert summaries[0].message_count == 1

    def test_scan_ignores_index_file(self, tmp_path: Path) -> None:
        # An index.jsonl should not appear as a session
        (tmp_path / "index.jsonl").write_text("{}\n", encoding="utf-8")
        (tmp_path / "real.jsonl").write_text(
            '{"type":"session_info","id":"m","timestamp":1.0}\n',
            encoding="utf-8",
        )

        mgr = SessionManager(tmp_path)
        # The index is malformed so catalog returns empty → falls back to scan
        summaries = mgr.list_sessions()
        assert len(summaries) == 1
        assert summaries[0].session_id == "real"


# ── get / latest_for_cwd ─────────────────────────────────────────────────


class TestLookup:
    @pytest.mark.asyncio
    async def test_get(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        session = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
        )
        assert mgr.get(session.session_id) is not None
        assert mgr.get("nonexistent") is None
        await session.shutdown()

    @pytest.mark.asyncio
    async def test_latest_for_cwd(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        await (await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            cwd="/project-a",
        )).shutdown()

        import time
        time.sleep(0.01)

        s2 = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            cwd="/project-a",
        )

        meta = mgr.latest_for_cwd("/project-a")
        assert meta is not None
        assert meta.session_id == s2.session_id

        assert mgr.latest_for_cwd("/project-b") is None
        await s2.shutdown()


# ── export ────────────────────────────────────────────────────────────────


class TestExport:
    @pytest.mark.asyncio
    async def test_export_jsonl(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        session = await mgr.create(
            provider=_make_provider(),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
        )
        sid = session.session_id
        await session.shutdown()

        text = await mgr.export(sid, "jsonl")
        assert text.strip()  # at least the session_info record

    @pytest.mark.asyncio
    async def test_export_nonexistent_raises(self, tmp_path: Path) -> None:
        mgr = SessionManager(tmp_path)
        with pytest.raises(FileNotFoundError):
            await mgr.export("nonexistent")
