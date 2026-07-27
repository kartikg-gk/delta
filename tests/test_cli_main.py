"""Tests for the CLI entry point — argument parsing, subcommands, and end-to-end flow."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from delta_app.cli.main import (
    _build_run_parser,
    _build_provider_parser,
    _build_session_parser,
    _resolve_model,
    _resolve_system,
    _sessions_dir,
    _session_path,
    get_version,
    main,
)
from delta_harness.contracts.transcript import ModelEntry, TextSegment
from delta_harness.provider.wire import StreamCloseEvent, StreamOpenEvent
from delta_model.scripted import ReplayProvider


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_get_version_returns_string():
    v = get_version()
    assert isinstance(v, str) and len(v) > 0


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------


def test_run_parser_defaults():
    ns = _build_run_parser().parse_args([])
    assert ns.prompt is None
    assert ns.print_mode is False
    assert ns.model is None
    assert ns.provider is None
    assert ns.resume is None
    assert ns.verbose is False
    assert ns.no_session is False


def test_run_parser_one_shot():
    ns = _build_run_parser().parse_args(["hello", "-p", "-m", "test-model"])
    assert ns.prompt == "hello"
    assert ns.print_mode is True
    assert ns.model == "test-model"


def test_provider_parser_list():
    ns = _build_provider_parser().parse_args(["list"])
    assert ns.action == "list"


def test_provider_parser_select():
    ns = _build_provider_parser().parse_args(["select", "openai"])
    assert ns.action == "select"
    assert ns.name == "openai"


def test_session_parser_list():
    ns = _build_session_parser().parse_args(["list"])
    assert ns.action == "list"


def test_session_parser_export():
    ns = _build_session_parser().parse_args(["export", "abc123", "-f", "text"])
    assert ns.action == "export"
    assert ns.session_id == "abc123"
    assert ns.format == "text"


# ---------------------------------------------------------------------------
# Resolver helpers
# ---------------------------------------------------------------------------


def test_resolve_model_default():
    assert _resolve_model(None) == "claude-sonnet-4-20250514"


def test_resolve_model_override():
    assert _resolve_model("gpt-4") == "gpt-4"


def test_resolve_model_env(monkeypatch):
    monkeypatch.setenv("DELTA_MODEL", "env-model")
    assert _resolve_model(None) == "env-model"


def test_resolve_system_default():
    result = _resolve_system(None)
    assert isinstance(result, str) and len(result) > 0


def test_resolve_system_override():
    assert _resolve_system("custom prompt") == "custom prompt"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_sessions_dir_default():
    p = _sessions_dir(None)
    assert p == Path.home() / ".delta" / "sessions"


def test_sessions_dir_override():
    p = _sessions_dir("/tmp/my-sessions")
    assert p == Path("/tmp/my-sessions")


def test_sessions_dir_env(monkeypatch):
    monkeypatch.setenv("DELTA_SESSIONS_DIR", "/from/env")
    p = _sessions_dir(None)
    assert p == Path("/from/env")


def test_session_path():
    base = Path("/sessions")
    assert _session_path(base, "abc") == Path("/sessions/abc.jsonl")


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------


def test_main_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "delta" in out


def test_main_provider_list(capsys):
    code = main(["provider", "list"])
    assert code == 0
    err = capsys.readouterr().err
    assert "anthropic" in err


def test_main_session_list_empty():
    with tempfile.TemporaryDirectory() as td:
        code = main(["session", "--session-dir", td, "list"])
        assert code == 0


def test_main_session_list_session_dir_after_subcommand():
    """--session-dir works when placed after the subcommand."""
    with tempfile.TemporaryDirectory() as td:
        code = main(["session", "list", "--session-dir", td])
        assert code == 0


# ---------------------------------------------------------------------------
# End-to-end one-shot flow with ReplayProvider
# ---------------------------------------------------------------------------


def _make_simple_reply(text: str = "Hello from Delta!") -> list[tuple]:
    """Build a single-turn wire event stream for ReplayProvider."""
    entry = ModelEntry(
        model="test-model",
        content=[TextSegment(text=text)],
        stop_reason="stop",
    )
    return [(
        StreamOpenEvent(partial=entry),
        StreamCloseEvent(reason="stop", message=entry),
    )]


def test_one_shot_print_mode(capsys):
    """One-shot --print mode: prompt in, streamed text out, exit 0."""
    provider = ReplayProvider(_make_simple_reply("Test reply."))

    with (
        patch("delta_app.cli.main._resolve_provider", return_value=provider),
        patch("delta_app.cli.main._load_tools", return_value=[]),
        patch("delta_app.cli.main._load_extensions", return_value=[]),
        patch("delta_app.cli.main._install_hooks"),
    ):
        code = main(["--no-session", "-p", "say hi"])

    assert code == 0
    out = capsys.readouterr().out
    assert "Test reply." in out


def test_one_shot_with_session():
    """One-shot with session persistence creates a JSONL file."""
    provider = ReplayProvider(_make_simple_reply())

    with tempfile.TemporaryDirectory() as td:
        with (
            patch("delta_app.cli.main._resolve_provider", return_value=provider),
            patch("delta_app.cli.main._load_tools", return_value=[]),
            patch("delta_app.cli.main._load_extensions", return_value=[]),
            patch("delta_app.cli.main._install_hooks"),
        ):
            code = main(["-p", "--session-dir", td, "test prompt"])

        assert code == 0
        # Should have created exactly one session .jsonl file (plus the catalog index)
        files = [p for p in Path(td).glob("*.jsonl") if p.name != "index.jsonl"]
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "session_info" in content
        assert "message" in content
        # CodingSession also maintains a catalog index
        assert (Path(td) / "index.jsonl").exists()


def test_one_shot_no_provider(capsys):
    """Missing provider should exit with an error."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--no-session", "-p", "--provider", "nonexistent", "hello"])
    assert exc_info.value.code != 0


def test_print_mode_requires_prompt(capsys):
    """--print without a prompt and stdin is a tty should error."""
    provider = ReplayProvider(_make_simple_reply())

    with (
        patch("delta_app.cli.main._resolve_provider", return_value=provider),
        patch("delta_app.cli.main._load_tools", return_value=[]),
        patch("delta_app.cli.main._load_extensions", return_value=[]),
        patch("delta_app.cli.main._install_hooks"),
        patch("sys.stdin") as mock_stdin,
    ):
        mock_stdin.isatty.return_value = True
        with pytest.raises(SystemExit):
            main(["--no-session", "-p"])


# ---------------------------------------------------------------------------
# Session export
# ---------------------------------------------------------------------------


def test_session_export_jsonl(capsys):
    """Export a session that was created by a one-shot run."""
    provider = ReplayProvider(_make_simple_reply("Export me."))

    with tempfile.TemporaryDirectory() as td:
        with (
            patch("delta_app.cli.main._resolve_provider", return_value=provider),
            patch("delta_app.cli.main._load_tools", return_value=[]),
            patch("delta_app.cli.main._load_extensions", return_value=[]),
            patch("delta_app.cli.main._install_hooks"),
        ):
            main(["-p", "--session-dir", td, "test"])

        files = list(Path(td).glob("*.jsonl"))
        assert files
        sid = files[0].stem

        code = main(["session", "--session-dir", td, "export", sid, "-f", "text"])
        assert code == 0
        out = capsys.readouterr().out
        assert "[user]" in out or "[assistant]" in out


def test_session_export_missing():
    """Exporting a non-existent session should error."""
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(SystemExit):
            main(["session", "--session-dir", td, "export", "nonexistent"])
