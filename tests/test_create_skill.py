"""`/create-skill <name> <body>` — authoring a skill from the composer."""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, str(__file__.rsplit("test_create_skill.py", 1)[0]))

from _session_helpers import _make_session, _text_turn  # noqa: E402

from delta_app.conversation import (  # noqa: E402
    derive_skill_description,
    parse_create_skill_arg,
    render_skill_file,
)
from delta_app.skillset import load_skills  # noqa: E402


async def _session(tmp_path):
    session = await _make_session(tmp_path, [_text_turn(["ok"])] * 6)
    session._cwd = str(tmp_path)
    session._skills_loader = lambda: load_skills([tmp_path / ".delta" / "skills"])
    session.reload_skills()
    return session


def _skill_file(tmp_path, name: str):
    return tmp_path / ".delta" / "skills" / name / "SKILL.md"


# ── parsing ────────────────────────────────────────────────────────────────


class TestParsing:
    @pytest.mark.parametrize("arg,expected", [
        ("", ("", "")),
        ("   ", ("", "")),
        ("review", ("review", "")),
        ("review do the thing", ("review", "do the thing")),
        ("  review   do it  ", ("review", "do it")),
    ])
    def test_split_name_and_body(self, arg, expected):
        assert parse_create_skill_arg(arg) == expected

    def test_body_keeps_newlines(self):
        name, body = parse_create_skill_arg("review line one\nline two")
        assert name == "review"
        assert body == "line one\nline two"

    @pytest.mark.parametrize("body,expected", [
        ("First line.\nSecond.", "First line."),
        ("\n\n  Indented first  \nmore", "Indented first"),
        ("# Heading\nbody", "Heading"),
        ("", ""),
    ])
    def test_description_derivation(self, body, expected):
        assert derive_skill_description(body) == expected

    def test_render_has_front_matter(self):
        text = render_skill_file("review", "Do reviews", "Body here")
        assert text.startswith("---\nname: review\ndescription: Do reviews\n---\n")
        assert text.rstrip().endswith("Body here")


# ── writing ────────────────────────────────────────────────────────────────


class TestCreate:
    async def test_creates_and_loads(self, tmp_path):
        session = await _session(tmp_path)
        out = await session.handle_command(
            "/create-skill review Review the diff for correctness bugs."
        )
        await session.shutdown()

        assert "Created skill 'review'" in out
        assert "/skill:review" in out
        assert _skill_file(tmp_path, "review").exists()

    async def test_new_skill_is_immediately_available(self, tmp_path):
        session = await _session(tmp_path)
        await session.handle_command("/create-skill review Check for bugs.")
        listing = await session.handle_command("/skills")
        await session.shutdown()
        assert "/skill:review" in listing
        assert "Check for bugs." in listing

    async def test_multiline_body_preserved(self, tmp_path):
        session = await _session(tmp_path)
        await session.handle_command(
            "/create-skill review First line.\nSecond line.\nThird line."
        )
        text = _skill_file(tmp_path, "review").read_text(encoding="utf-8")
        await session.shutdown()
        assert "Second line." in text
        assert "Third line." in text

    async def test_description_taken_from_first_line(self, tmp_path):
        session = await _session(tmp_path)
        await session.handle_command("/create-skill review Summary line.\nDetail.")
        text = _skill_file(tmp_path, "review").read_text(encoding="utf-8")
        await session.shutdown()
        assert "description: Summary line." in text

    async def test_overwrite_reports_update(self, tmp_path):
        session = await _session(tmp_path)
        await session.handle_command("/create-skill review Original.")
        out = await session.handle_command("/create-skill review Replacement.")
        text = _skill_file(tmp_path, "review").read_text(encoding="utf-8")
        await session.shutdown()
        assert "Updated skill" in out
        assert "Replacement." in text
        assert "Original." not in text

    async def test_invocation_uses_the_new_body(self, tmp_path):
        session = await _session(tmp_path)
        await session.handle_command(
            "/create-skill review Hunt for off-by-one errors."
        )
        block = session.expand_skill("/skill:review the parser")
        await session.shutdown()
        assert block is not None
        assert "Hunt for off-by-one errors." in block


# ── validation ─────────────────────────────────────────────────────────────


class TestValidation:
    async def test_missing_arguments_shows_usage(self, tmp_path):
        session = await _session(tmp_path)
        out = await session.handle_command("/create-skill")
        await session.shutdown()
        assert "Usage:" in out

    async def test_missing_body_rejected(self, tmp_path):
        session = await _session(tmp_path)
        out = await session.handle_command("/create-skill lonely")
        await session.shutdown()
        assert "Nothing to save" in out
        assert not _skill_file(tmp_path, "lonely").exists()

    @pytest.mark.parametrize("name", ["../evil", "a/b", "a\\b", "-lead", ".."])
    async def test_unsafe_names_rejected(self, tmp_path, name):
        session = await _session(tmp_path)
        out = await session.handle_command(f"/create-skill {name} some body")
        await session.shutdown()
        assert "Invalid skill name" in out

    async def test_traversal_writes_nothing_outside_project(self, tmp_path):
        session = await _session(tmp_path)
        await session.handle_command("/create-skill ../escaped body text")
        await session.shutdown()
        assert not (tmp_path.parent / "escaped").exists()


class TestRegistration:
    def test_command_is_registered(self):
        from delta_app.conversation import CodingSession

        assert any(name == "create-skill" for name, _ in CodingSession.COMMANDS)

    def test_autocomplete_offers_it(self):
        pytest.importorskip("textual")
        from delta_app.conversation import CodingSession
        from delta_app.tui.completions import match_commands

        names = [c.name for c in match_commands("/create", CodingSession.COMMANDS)]
        assert "create-skill" in names
