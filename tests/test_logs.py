"""Run logging: .logs/<session>/{trajectory.json,events.jsonl,stdout.log}."""

from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, str(__file__.rsplit("test_logs.py", 1)[0]))

from _session_helpers import _make_session, _text_turn, _tool_turn  # noqa: E402

from delta_app.logs import RunLogger, log_root, logs_enabled  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("DELTA_LOG_DIR", "DELTA_NO_LOGS"):
        monkeypatch.delenv(name, raising=False)


# ── configuration ──────────────────────────────────────────────────────────


class TestConfig:
    def test_default_location(self, tmp_path):
        assert log_root(tmp_path) == tmp_path / ".logs"

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DELTA_LOG_DIR", str(tmp_path / "custom"))
        assert log_root(tmp_path) == tmp_path / "custom"

    @pytest.mark.parametrize("value,expected", [
        ("1", False), ("true", False), ("yes", False),
        ("0", True), ("", True),
    ])
    def test_disable_switch(self, monkeypatch, value, expected):
        monkeypatch.setenv("DELTA_NO_LOGS", value)
        assert logs_enabled() is expected

    def test_create_returns_none_when_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DELTA_NO_LOGS", "1")
        assert RunLogger.create("s1", root=tmp_path) is None


# ── artefacts ──────────────────────────────────────────────────────────────


class TestArtefacts:
    async def test_writes_all_three_files(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["hi"])] * 4)
        logger = RunLogger.create(
            session.session_id, provider="replay", model="m",
            root=tmp_path / ".logs",
        )
        logger.stats_source = lambda: session.usage
        session.harness.on_event(logger.record_event)

        async for _ in session.submit("hello"):
            pass
        await session.shutdown()

        d = tmp_path / ".logs" / session.session_id
        assert {p.name for p in d.iterdir()} == {
            "trajectory.json", "events.jsonl", "stdout.log",
        }

    async def test_trajectory_captures_tool_use(self, tmp_path):
        target = tmp_path / "made.py"
        session = await _make_session(
            tmp_path, [_tool_turn(str(target)), _text_turn(["done"])],
        )
        attach_logger_to(session, tmp_path)

        async for _ in session.submit("create it"):
            pass
        await session.shutdown()

        doc = _trajectory(tmp_path, session.session_id)
        kinds = [s["kind"] for s in doc["steps"]]
        assert kinds == ["prompt", "tool_call", "tool_result", "assistant"]
        assert doc["steps"][1]["tool"] == "Write"
        assert doc["steps"][2]["is_error"] is False

    async def test_trajectory_has_usage_totals(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["hi"])] * 4)
        attach_logger_to(session, tmp_path)
        async for _ in session.submit("hello"):
            pass
        await session.shutdown()

        doc = _trajectory(tmp_path, session.session_id)
        assert doc["usage"]["turns"] >= 1
        assert doc["session_id"] == session.session_id

    async def test_events_are_one_json_object_per_line(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["hi"])] * 4)
        attach_logger_to(session, tmp_path)
        async for _ in session.submit("hello"):
            pass
        await session.shutdown()

        path = tmp_path / ".logs" / session.session_id / "events.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        for line in lines:
            assert "type" in json.loads(line)

    async def test_trajectory_written_without_explicit_shutdown(self, tmp_path):
        """A killed run must still leave a readable trajectory."""
        session = await _make_session(tmp_path, [_text_turn(["hi"])] * 4)
        attach_logger_to(session, tmp_path)
        async for _ in session.submit("hello"):
            pass
        # no shutdown() call
        doc = _trajectory(tmp_path, session.session_id)
        assert doc["steps"]
        await session.shutdown()


# ── safety ─────────────────────────────────────────────────────────────────


class TestScrubbing:
    def test_secrets_removed_from_stdout(self, tmp_path):
        logger = RunLogger.create("s1", root=tmp_path)
        logger.write_line("connecting with api_key=sk-abcdef1234567890")
        text = (tmp_path / "s1" / "stdout.log").read_text(encoding="utf-8")
        assert "sk-abcdef1234567890" not in text
        assert "REDACTED" in text

    def test_secrets_removed_from_trajectory(self, tmp_path):
        logger = RunLogger.create("s2", root=tmp_path)
        logger.steps.append({"kind": "x", "text": "api_key=sk-topsecret999999"})
        logger.finish()
        text = (tmp_path / "s2" / "trajectory.json").read_text(encoding="utf-8")
        assert "sk-topsecret999999" not in text


class TestResilience:
    def test_unwritable_root_returns_none(self, tmp_path):
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        assert RunLogger.create("s1", root=blocker) is None

    def test_finish_is_repeatable(self, tmp_path):
        logger = RunLogger.create("s3", root=tmp_path)
        logger.finish()
        logger.steps.append({"kind": "prompt", "text": "later"})
        logger.finish()
        doc = json.loads(
            (tmp_path / "s3" / "trajectory.json").read_text(encoding="utf-8")
        )
        assert len(doc["steps"]) == 1


# ── helpers ────────────────────────────────────────────────────────────────


def attach_logger_to(session, tmp_path) -> None:
    logger = RunLogger.create(
        session.session_id, provider="replay", model="m",
        root=tmp_path / ".logs",
    )
    logger.stats_source = lambda: session.usage
    session.harness.on_event(logger.record_event)


def _trajectory(tmp_path, session_id) -> dict:
    path = tmp_path / ".logs" / session_id / "trajectory.json"
    return json.loads(path.read_text(encoding="utf-8"))
