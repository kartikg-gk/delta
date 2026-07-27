"""Tests for SessionCatalog and SessionMeta — persistent session indexing."""

from __future__ import annotations

import json
from pathlib import Path
from time import time

import pytest

from delta_harness.session.index import (
    SessionCatalog,
    SessionMeta,
    SessionMetaWire,
    _wire_to_meta,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _make_meta(
    session_id: str = "abc123",
    vault_path: str = "/sessions/abc123.jsonl",
    cwd: str = "/project",
    model: str = "test-model",
    provider: str = "test",
    title: str | None = None,
    created_at: float | None = None,
    updated_at: float | None = None,
) -> SessionMeta:
    now = time()
    return SessionMeta(
        session_id=session_id,
        vault_path=vault_path,
        cwd=cwd,
        model=model,
        provider=provider,
        title=title,
        created_at=created_at or now,
        updated_at=updated_at or now,
    )


# ── SessionMeta / SessionMetaWire ─────────────────────────────────────────


class TestSessionMeta:
    def test_frozen(self) -> None:
        meta = _make_meta()
        with pytest.raises(AttributeError):
            meta.title = "changed"  # type: ignore[misc]

    def test_to_wire_roundtrip(self) -> None:
        meta = _make_meta(title="My Session")
        wire = meta.to_wire()
        back = _wire_to_meta(wire)
        assert back.session_id == meta.session_id
        assert back.vault_path == meta.vault_path
        assert back.cwd == meta.cwd
        assert back.model == meta.model
        assert back.provider == meta.provider
        assert back.title == meta.title
        assert back.created_at == meta.created_at
        assert back.updated_at == meta.updated_at

    def test_wire_serializes_to_json(self) -> None:
        meta = _make_meta()
        wire = meta.to_wire()
        raw = json.loads(wire.model_dump_json())
        assert raw["session_id"] == "abc123"
        assert "title" in raw or raw.get("title") is None

    def test_wire_excludes_none_title(self) -> None:
        meta = _make_meta(title=None)
        wire = meta.to_wire()
        raw = json.loads(wire.model_dump_json(exclude_none=True))
        assert "title" not in raw


# ── SessionCatalog.prepare ─────────────────────────────────────────────────


class TestPrepare:
    def test_prepare_sets_timestamps(self) -> None:
        before = time()
        meta = SessionCatalog.prepare(
            session_id="s1",
            vault_path="/v/s1.jsonl",
            cwd="/project",
            model="m1",
            provider="p1",
        )
        after = time()
        assert before <= meta.created_at <= after
        assert before <= meta.updated_at <= after
        assert meta.title is None

    def test_prepare_with_title(self) -> None:
        meta = SessionCatalog.prepare(
            session_id="s1",
            vault_path="/v/s1.jsonl",
            cwd="/project",
            model="m1",
            provider="p1",
            title="My Session",
        )
        assert meta.title == "My Session"


# ── SessionCatalog: basic operations ───────────────────────────────────────


class TestCatalogBasics:
    def test_empty_catalog(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        assert catalog.list_all() == []
        assert catalog.get("nonexistent") is None
        assert catalog.latest_for_cwd("/any") is None

    def test_upsert_and_get(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        meta = _make_meta(session_id="s1")
        catalog.upsert(meta)

        found = catalog.get("s1")
        assert found is not None
        assert found.session_id == "s1"
        assert found.cwd == "/project"

    def test_upsert_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "dir"
        catalog = SessionCatalog(nested)
        meta = _make_meta()
        catalog.upsert(meta)
        assert catalog.path.exists()

    def test_list_all_returns_sorted(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        now = time()
        catalog.upsert(_make_meta(session_id="old", updated_at=now - 100))
        catalog.upsert(_make_meta(session_id="new", updated_at=now))
        catalog.upsert(_make_meta(session_id="mid", updated_at=now - 50))

        result = catalog.list_all()
        ids = [m.session_id for m in result]
        assert ids == ["new", "mid", "old"]

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        catalog.upsert(_make_meta(session_id="s1"))
        assert catalog.get("s999") is None


# ── SessionCatalog: deduplication ──────────────────────────────────────────


class TestDeduplication:
    def test_keeps_newest_updated_at(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        now = time()
        catalog.upsert(_make_meta(session_id="s1", title="Old", updated_at=now - 10))
        catalog.upsert(_make_meta(session_id="s1", title="New", updated_at=now))

        found = catalog.get("s1")
        assert found is not None
        assert found.title == "New"

    def test_dedup_does_not_count_duplicates_in_list(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        now = time()
        catalog.upsert(_make_meta(session_id="s1", updated_at=now))
        catalog.upsert(_make_meta(session_id="s1", updated_at=now + 1))
        catalog.upsert(_make_meta(session_id="s2", updated_at=now))

        result = catalog.list_all()
        assert len(result) == 2

    def test_older_duplicate_ignored(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        now = time()
        catalog.upsert(_make_meta(session_id="s1", title="Newer", updated_at=now))
        catalog.upsert(_make_meta(session_id="s1", title="Older", updated_at=now - 5))

        found = catalog.get("s1")
        assert found is not None
        assert found.title == "Newer"


# ── SessionCatalog: touch ─────────────────────────────────────────────────


class TestTouch:
    def test_touch_updates_timestamp(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        old_time = time() - 100
        catalog.upsert(_make_meta(session_id="s1", updated_at=old_time))

        before = time()
        result = catalog.touch("s1")
        after = time()

        assert result is not None
        assert before <= result.updated_at <= after
        assert result.updated_at > old_time

    def test_touch_preserves_other_fields(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        catalog.upsert(_make_meta(
            session_id="s1",
            title="Keep This",
            model="original-model",
            created_at=1000.0,
        ))

        result = catalog.touch("s1")
        assert result is not None
        assert result.title == "Keep This"
        assert result.model == "original-model"
        assert result.created_at == 1000.0

    def test_touch_missing_returns_none(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        assert catalog.touch("nonexistent") is None


# ── SessionCatalog: latest_for_cwd ────────────────────────────────────────


class TestLatestForCwd:
    def test_finds_latest(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        now = time()
        catalog.upsert(_make_meta(session_id="s1", cwd="/a", updated_at=now - 10))
        catalog.upsert(_make_meta(session_id="s2", cwd="/a", updated_at=now))
        catalog.upsert(_make_meta(session_id="s3", cwd="/b", updated_at=now + 10))

        result = catalog.latest_for_cwd("/a")
        assert result is not None
        assert result.session_id == "s2"

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        catalog.upsert(_make_meta(session_id="s1", cwd="/a"))
        assert catalog.latest_for_cwd("/b") is None

    def test_empty_catalog_returns_none(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        assert catalog.latest_for_cwd("/any") is None


# ── SessionCatalog: error resilience ───────────────────────────────────────


class TestErrorResilience:
    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        catalog.upsert(_make_meta(session_id="good"))

        # Inject malformed lines directly
        with catalog.path.open("a", encoding="utf-8") as fh:
            fh.write("not valid json\n")
            fh.write('{"session_id": "incomplete"}\n')  # missing required fields
            fh.write("\n")  # blank line

        result = catalog.list_all()
        assert len(result) == 1
        assert result[0].session_id == "good"

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path / "nonexistent")
        assert catalog.list_all() == []
        assert catalog.get("any") is None

    def test_path_property(self, tmp_path: Path) -> None:
        catalog = SessionCatalog(tmp_path)
        assert catalog.path == tmp_path / "index.jsonl"


# ── Integration with CodingSession ────────────────────────────────────────


class TestCodingSessionIntegration:
    """Verify that CodingSession.create and .resume interact with the catalog."""

    @pytest.mark.asyncio
    async def test_create_indexes_session(self, tmp_path: Path) -> None:
        from delta_app.conversation import CodingSession
        from delta_model.scripted import ReplayProvider
        from delta_harness.provider.wire import StreamCloseEvent
        from delta_harness.contracts.transcript import ModelEntry, TextSegment
        from delta_harness.contracts.transcript.diagnostics import (
            CostBreakdown,
            UsageStats,
        )

        reply = ModelEntry(
            model="test-model",
            content=[TextSegment(text="Hello!")],
            stop_reason="stop",
            usage=UsageStats(
                total_tokens=150, input=100, output=50,
                cost=CostBreakdown(total=0.001),
            ),
        )
        provider = ReplayProvider(
            [[StreamCloseEvent(reason="stop", message=reply)]]
        )

        session = await CodingSession.create(
            provider=provider,
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            sessions_dir=tmp_path,
        )

        # Catalog should contain the session
        catalog = SessionCatalog(tmp_path)
        meta = catalog.get(session.session_id)
        assert meta is not None
        assert meta.model == "test-model"
        assert meta.provider == "test"
        assert meta.title is None
        await session.shutdown()

    @pytest.mark.asyncio
    async def test_resume_touches_catalog(self, tmp_path: Path) -> None:
        from delta_app.conversation import CodingSession
        from delta_model.scripted import ReplayProvider
        from delta_harness.provider.wire import StreamCloseEvent
        from delta_harness.contracts.transcript import ModelEntry, TextSegment
        from delta_harness.contracts.transcript.diagnostics import (
            CostBreakdown,
            UsageStats,
        )

        reply = ModelEntry(
            model="test-model",
            content=[TextSegment(text="Hello!")],
            stop_reason="stop",
            usage=UsageStats(
                total_tokens=150, input=100, output=50,
                cost=CostBreakdown(total=0.001),
            ),
        )
        provider = ReplayProvider(
            [[StreamCloseEvent(reason="stop", message=reply)]]
        )

        session = await CodingSession.create(
            provider=provider,
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            sessions_dir=tmp_path,
        )
        sid = session.session_id

        # Record original timestamp
        catalog = SessionCatalog(tmp_path)
        original = catalog.get(sid)
        assert original is not None
        original_ts = original.updated_at

        await session.shutdown()

        # Resume — should touch the catalog
        import time as _time
        _time.sleep(0.01)  # ensure time difference

        provider2 = ReplayProvider(
            [[StreamCloseEvent(reason="stop", message=reply)]]
        )
        resumed = await CodingSession.resume(
            sid,
            provider=provider2,
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            sessions_dir=tmp_path,
        )

        refreshed = catalog.get(sid)
        assert refreshed is not None
        assert refreshed.updated_at >= original_ts
        await resumed.shutdown()

    @pytest.mark.asyncio
    async def test_cmd_name_syncs_to_catalog(self, tmp_path: Path) -> None:
        from delta_app.conversation import CodingSession
        from delta_model.scripted import ReplayProvider
        from delta_harness.provider.wire import StreamCloseEvent
        from delta_harness.contracts.transcript import ModelEntry, TextSegment
        from delta_harness.contracts.transcript.diagnostics import (
            CostBreakdown,
            UsageStats,
        )

        reply = ModelEntry(
            model="test-model",
            content=[TextSegment(text="Hello!")],
            stop_reason="stop",
            usage=UsageStats(
                total_tokens=150, input=100, output=50,
                cost=CostBreakdown(total=0.001),
            ),
        )
        provider = ReplayProvider(
            [[StreamCloseEvent(reason="stop", message=reply)]]
        )

        session = await CodingSession.create(
            provider=provider,
            provider_name="test",
            model="test-model",
            system="You are helpful.",
            sessions_dir=tmp_path,
        )

        await session.handle_command("/name Renamed Session")

        catalog = SessionCatalog(tmp_path)
        meta = catalog.get(session.session_id)
        assert meta is not None
        assert meta.title == "Renamed Session"
        await session.shutdown()

    @pytest.mark.asyncio
    async def test_ephemeral_session_has_no_catalog(self) -> None:
        from delta_app.conversation import CodingSession
        from delta_model.scripted import ReplayProvider

        session = await CodingSession.create(
            provider=ReplayProvider([]),
            provider_name="test",
            model="test-model",
            system="You are helpful.",
        )

        assert session.catalog is None
        await session.shutdown()
