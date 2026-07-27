"""Tests for reload.py — diagnostic context, logger, and secret scrubbing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from delta_app.reload import (
    ReloadDiagnosticContext,
    ReloadDiagnosticLogger,
    build_base_entry,
    build_exception_entry,
    build_failure_entry,
    new_run_id,
)


# ── new_run_id ───────────────────────────────────────────────────────────


class TestNewRunId:
    def test_length(self) -> None:
        assert len(new_run_id()) == 12

    def test_hex_chars(self) -> None:
        rid = new_run_id()
        assert all(c in "0123456789abcdef" for c in rid)

    def test_unique(self) -> None:
        ids = {new_run_id() for _ in range(50)}
        assert len(ids) == 50


# ── ReloadDiagnosticContext ──────────────────────────────────────────────


class TestReloadDiagnosticContext:
    def test_frozen(self) -> None:
        ctx = ReloadDiagnosticContext.create()
        with pytest.raises(AttributeError):
            ctx.run_id = "x"  # type: ignore[misc]

    def test_create_unique_ids(self) -> None:
        ids = {ReloadDiagnosticContext.create().run_id for _ in range(20)}
        assert len(ids) == 20

    def test_run_id_length(self) -> None:
        assert len(ReloadDiagnosticContext.create().run_id) == 12

    def test_started_at_is_iso_utc(self) -> None:
        ctx = ReloadDiagnosticContext.create()
        assert ctx.started_at.endswith("Z")
        assert "T" in ctx.started_at

    def test_manual_construction(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="abc", started_at="2026-01-01T00:00:00Z")
        assert ctx.run_id == "abc"
        assert ctx.started_at == "2026-01-01T00:00:00Z"


# ── build_base_entry ────────────────────────────────────────────────────


class TestBuildBaseEntry:
    def test_contains_run_id(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="run1", started_at="t")
        entry = build_base_entry(ctx)
        assert entry["run_id"] == "run1"

    def test_contains_ts(self) -> None:
        ctx = ReloadDiagnosticContext.create()
        entry = build_base_entry(ctx)
        assert entry["ts"].endswith("Z")

    def test_only_two_keys(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="r", started_at="t")
        entry = build_base_entry(ctx)
        assert set(entry.keys()) == {"run_id", "ts"}


# ── build_failure_entry ─────────────────────────────────────────────────


class TestBuildFailureEntry:
    def test_basic_fields(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="r1", started_at="t")
        entry = build_failure_entry(ctx, "skills", "bad", "not valid UTF-8")
        assert entry["run_id"] == "r1"
        assert entry["level"] == "failure"
        assert entry["category"] == "skills"
        assert entry["name"] == "bad"
        assert entry["error"] == "not valid UTF-8"
        assert "path" not in entry
        assert "traceback" not in entry

    def test_with_path(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="r", started_at="t")
        entry = build_failure_entry(ctx, "p", "x", "e", path="/a/b.md")
        assert entry["path"] == "/a/b.md"

    def test_scrubs_secrets(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="r", started_at="t")
        entry = build_failure_entry(ctx, "c", "k", "sk-secret12345678 leaked")
        assert "sk-secret" not in entry["error"]
        assert "[REDACTED]" in entry["error"]

    def test_truncates_long_error(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="r", started_at="t")
        entry = build_failure_entry(ctx, "s", "x", "x" * 500)
        assert len(entry["error"]) <= 301


# ── build_exception_entry ───────────────────────────────────────────────


class TestBuildExceptionEntry:
    def test_basic_fields(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="r1", started_at="t")
        try:
            raise ValueError("boom")
        except ValueError as exc:
            entry = build_exception_entry(ctx, "skills", "bad", exc)

        assert entry["level"] == "error"
        assert entry["error"] == "boom"
        assert isinstance(entry["traceback"], list)
        assert any("ValueError" in line for line in entry["traceback"])

    def test_with_path(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="r", started_at="t")
        try:
            raise RuntimeError("err")
        except RuntimeError as exc:
            entry = build_exception_entry(ctx, "p", "x", exc, path="/a.py")
        assert entry["path"] == "/a.py"

    def test_scrubs_error(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="r", started_at="t")
        try:
            raise ValueError("Bearer eyJtoken123")
        except ValueError as exc:
            entry = build_exception_entry(ctx, "c", "x", exc)
        assert "eyJtoken" not in entry["error"]

    def test_scrubs_traceback(self) -> None:
        ctx = ReloadDiagnosticContext(run_id="r", started_at="t")
        try:
            raise RuntimeError("Auth failed with sk-supersecret12345")
        except RuntimeError as exc:
            entry = build_exception_entry(ctx, "c", "auth", exc)
        for line in entry["traceback"]:
            assert "sk-supersecret" not in line


# ── ReloadDiagnosticLogger ──────────────────────────────────────────────


class TestReloadDiagnosticLogger:
    def test_log_failure_creates_file(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx = ReloadDiagnosticContext.create()
        logger.log_failure(ctx, "skills", "bad", "not valid UTF-8")
        assert logger.log_path.exists()
        assert logger.log_path.name == "reload_diag.jsonl"

    def test_log_failure_valid_json(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx = ReloadDiagnosticContext.create()
        logger.log_failure(ctx, "prompts", "broken.md", "parse error")
        parsed = json.loads(logger.log_path.read_text(encoding="utf-8").strip())
        assert parsed["level"] == "failure"
        assert parsed["category"] == "prompts"
        assert parsed["name"] == "broken.md"
        assert parsed["error"] == "parse error"
        assert parsed["run_id"] == ctx.run_id

    def test_log_failure_with_path(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx = ReloadDiagnosticContext.create()
        logger.log_failure(ctx, "skills", "x", "err", path="/s/x/SKILL.md")
        parsed = json.loads(logger.log_path.read_text(encoding="utf-8").strip())
        assert parsed["path"] == "/s/x/SKILL.md"

    def test_log_failure_scrubs_secrets(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx = ReloadDiagnosticContext.create()
        logger.log_failure(ctx, "config", "key", "sk-secret12345678 leaked")
        content = logger.log_path.read_text(encoding="utf-8")
        assert "sk-secret" not in content
        assert "[REDACTED]" in content

    def test_log_exception_includes_traceback(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx = ReloadDiagnosticContext.create()
        try:
            raise ValueError("something broke")
        except ValueError as exc:
            logger.log_exception(ctx, "skills", "bad-skill", exc)

        parsed = json.loads(logger.log_path.read_text(encoding="utf-8").strip())
        assert parsed["level"] == "error"
        assert parsed["error"] == "something broke"
        assert isinstance(parsed["traceback"], list)
        assert any("ValueError" in line for line in parsed["traceback"])

    def test_log_exception_scrubs_traceback(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx = ReloadDiagnosticContext.create()
        try:
            raise RuntimeError("Auth failed with sk-supersecret12345")
        except RuntimeError as exc:
            logger.log_exception(ctx, "config", "auth", exc)

        content = logger.log_path.read_text(encoding="utf-8")
        assert "sk-supersecret" not in content
        assert "[REDACTED]" in content

    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx = ReloadDiagnosticContext.create()
        logger.log_failure(ctx, "skills", "a", "err1")
        logger.log_failure(ctx, "prompts", "b", "err2")
        logger.log_failure(ctx, "config", "c", "err3")
        lines = logger.log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert parsed["run_id"] == ctx.run_id

    def test_creates_nested_directory(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "deep" / "nested" / "logs"
        logger = ReloadDiagnosticLogger(log_dir)
        ctx = ReloadDiagnosticContext.create()
        logger.log_failure(ctx, "s", "x", "e")
        assert logger.log_path.exists()

    def test_custom_filename(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path, filename="custom.jsonl")
        ctx = ReloadDiagnosticContext.create()
        logger.log_failure(ctx, "s", "x", "e")
        assert logger.log_path.name == "custom.jsonl"

    def test_compact_json(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx = ReloadDiagnosticContext.create()
        logger.log_failure(ctx, "s", "x", "e")
        line = logger.log_path.read_text(encoding="utf-8").strip()
        assert ": " not in line
        assert ", " not in line

    def test_entries_have_utc_timestamps(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx = ReloadDiagnosticContext.create()
        logger.log_failure(ctx, "s", "x", "e")
        parsed = json.loads(logger.log_path.read_text(encoding="utf-8").strip())
        assert parsed["ts"].endswith("Z")

    def test_log_path_property(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        assert logger.log_path == tmp_path / "reload_diag.jsonl"

    def test_separate_contexts_different_run_ids(self, tmp_path: Path) -> None:
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx1 = ReloadDiagnosticContext.create()
        ctx2 = ReloadDiagnosticContext.create()
        logger.log_failure(ctx1, "s", "a", "e1")
        logger.log_failure(ctx2, "s", "b", "e2")
        lines = logger.log_path.read_text(encoding="utf-8").strip().splitlines()
        ids = {json.loads(line)["run_id"] for line in lines}
        assert len(ids) == 2

    def test_no_content_leaked(self, tmp_path: Path) -> None:
        """Log entries must only contain expected metadata keys."""
        logger = ReloadDiagnosticLogger(tmp_path)
        ctx = ReloadDiagnosticContext.create()
        logger.log_failure(ctx, "skills", "my-skill", "file not found")
        parsed = json.loads(logger.log_path.read_text(encoding="utf-8").strip())
        allowed = {"run_id", "ts", "level", "category", "name", "error", "path", "traceback"}
        assert set(parsed.keys()) <= allowed


# ── scrubbing edge cases ─────────────────────────────────────────────────


class TestScrubbing:
    def _scrub_via_failure(self, message: str) -> str:
        """Round-trip through build_failure_entry to test scrubbing."""
        ctx = ReloadDiagnosticContext(run_id="r", started_at="t")
        return build_failure_entry(ctx, "x", "y", message)["error"]

    def test_scrub_api_key(self) -> None:
        assert "sk-abc" not in self._scrub_via_failure("key sk-abc123456789defgh")
        assert "[REDACTED]" in self._scrub_via_failure("key sk-abc123456789defgh")

    def test_scrub_bearer_token(self) -> None:
        result = self._scrub_via_failure("Bearer eyJhbGciOiJIUzI1NiJ9.tok")
        assert "eyJ" not in result

    def test_scrub_password(self) -> None:
        assert "mysecret" not in self._scrub_via_failure("password=mysecretpassword")

    def test_scrub_token_equals(self) -> None:
        assert "abc123" not in self._scrub_via_failure("token=abc123secret")

    def test_scrub_secret_colon(self) -> None:
        assert "mysecretvalue" not in self._scrub_via_failure("secret: mysecretvalue")

    def test_no_scrub_normal_message(self) -> None:
        assert self._scrub_via_failure("file not found") == "file not found"

    def test_multiple_secrets(self) -> None:
        msg = "key sk-aaaa11112222 and Bearer xyz.token.here"
        result = self._scrub_via_failure(msg)
        assert "sk-aaaa" not in result
        assert "xyz.token" not in result
        assert result.count("[REDACTED]") == 2

    def test_truncates_long_message(self) -> None:
        result = self._scrub_via_failure("x" * 500)
        assert len(result) <= 301
        assert result.endswith("…")


# ── integration ──────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_flow(self, tmp_path: Path) -> None:
        """Simulate logging during a reload: context → failures + exception → file."""
        ctx = ReloadDiagnosticContext.create()
        logger = ReloadDiagnosticLogger(tmp_path)

        # Expected failure
        logger.log_failure(ctx, "skills", "broken-skill", "not valid UTF-8",
                           path="/skills/broken/SKILL.md")

        # Unexpected exception
        try:
            raise RuntimeError("disk full")
        except RuntimeError as exc:
            logger.log_exception(ctx, "prompts", "bad.md", exc,
                                 path="/prompts/bad.md")

        lines = logger.log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        failure = json.loads(lines[0])
        assert failure["level"] == "failure"
        assert failure["run_id"] == ctx.run_id
        assert failure["category"] == "skills"
        assert failure["path"] == "/skills/broken/SKILL.md"

        error = json.loads(lines[1])
        assert error["level"] == "error"
        assert error["run_id"] == ctx.run_id
        assert "traceback" in error
        assert any("RuntimeError" in line for line in error["traceback"])
