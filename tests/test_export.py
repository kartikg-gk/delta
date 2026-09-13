"""`/export` writing transcripts to a user-supplied path."""

from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, str(__file__.rsplit("test_export.py", 1)[0]))

from _session_helpers import _make_session, _text_turn  # noqa: E402

from delta_app.conversation import parse_export_arg  # noqa: E402


async def _session_with_history(tmp_path):
    session = await _make_session(tmp_path, [_text_turn(["hello there"])] * 4)
    async for _ in session.submit("first prompt"):
        pass
    return session


# ── argument parsing ───────────────────────────────────────────────────────


class TestParseArg:
    @pytest.mark.parametrize("arg,expected", [
        ("", ("text", None)),
        ("json", ("json", None)),
        ("jsonl", ("jsonl", None)),
        ("text", ("text", None)),
    ])
    def test_format_only(self, arg, expected):
        assert parse_export_arg(arg) == expected

    @pytest.mark.parametrize("arg,expected", [
        ("out.json", ("json", "out.json")),
        ("dump.jsonl", ("jsonl", "dump.jsonl")),
        ("notes.txt", ("text", "notes.txt")),
    ])
    def test_format_inferred_from_extension(self, arg, expected):
        assert parse_export_arg(arg) == expected

    @pytest.mark.parametrize("arg,expected", [
        ("json out.json", ("json", "out.json")),
        ("jsonl ~/a/b.jsonl", ("jsonl", "~/a/b.jsonl")),
        ("text ./exports/", ("text", "./exports/")),
    ])
    def test_explicit_format_and_path(self, arg, expected):
        assert parse_export_arg(arg) == expected

    def test_bare_word_is_a_format_not_a_path(self):
        """A typo must not be written to disk as a filename."""
        assert parse_export_arg("nope") == ("nope", None)

    def test_relative_dir_is_a_path(self):
        assert parse_export_arg("./exports") == ("text", "./exports")


# ── writing ────────────────────────────────────────────────────────────────


class TestExportToPath:
    async def test_no_path_returns_content(self, tmp_path):
        session = await _session_with_history(tmp_path)
        out = await session.handle_command("/export")
        await session.shutdown()
        assert "first prompt" in out

    async def test_writes_explicit_file(self, tmp_path):
        session = await _session_with_history(tmp_path)
        target = tmp_path / "a.json"
        msg = await session.handle_command(f"/export json {target}")
        await session.shutdown()
        assert "Exported json" in msg
        assert target.exists()
        json.loads(target.read_text(encoding="utf-8"))

    async def test_infers_format_from_path(self, tmp_path):
        session = await _session_with_history(tmp_path)
        target = tmp_path / "b.jsonl"
        msg = await session.handle_command(f"/export {target}")
        await session.shutdown()
        assert "Exported jsonl" in msg
        assert target.read_text(encoding="utf-8").strip()

    async def test_creates_missing_directories(self, tmp_path):
        session = await _session_with_history(tmp_path)
        target = tmp_path / "deep" / "nested" / "c.txt"
        await session.handle_command(f"/export {target}")
        await session.shutdown()
        assert target.exists()

    async def test_directory_destination_names_file_after_session(self, tmp_path):
        session = await _session_with_history(tmp_path)
        outdir = tmp_path / "outdir"
        outdir.mkdir()
        msg = await session.handle_command(f"/export json {outdir}")
        sid = session.session_id
        await session.shutdown()
        assert (outdir / f"{sid}.json").exists()
        assert sid in msg

    async def test_trailing_slash_treated_as_directory(self, tmp_path):
        session = await _session_with_history(tmp_path)
        sid = session.session_id
        await session.handle_command(f"/export json {tmp_path}/fresh/")
        await session.shutdown()
        assert (tmp_path / "fresh" / f"{sid}.json").exists()

    async def test_relative_path_resolves_against_session_cwd(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["hi"])] * 4)
        async for _ in session.submit("p"):
            pass
        cwd = session.cwd
        await session.handle_command("/export rel-out.txt")
        await session.shutdown()
        from pathlib import Path

        assert (Path(cwd) / "rel-out.txt").exists()
        (Path(cwd) / "rel-out.txt").unlink()

    async def test_unknown_format_rejected_without_writing(self, tmp_path):
        session = await _session_with_history(tmp_path)
        msg = await session.handle_command("/export nope")
        await session.shutdown()
        assert "Unknown format" in msg
        assert not (tmp_path / "nope").exists()

    async def test_write_failure_reported_not_raised(self, tmp_path):
        session = await _session_with_history(tmp_path)
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file", encoding="utf-8")
        msg = await session.handle_command(f"/export json {blocker}/inside.json")
        await session.shutdown()
        assert "Could not write export" in msg
