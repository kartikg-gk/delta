"""`/skills` listing, viewing, reloading, and `/skill:<name>` invocation."""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, str(__file__.rsplit("test_skills_command.py", 1)[0]))

from _session_helpers import _make_session, _text_turn  # noqa: E402

from delta_app.skillset import load_skills  # noqa: E402


def _write_skill(root, name: str, description: str, body: str = "") -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n"
        f"{body or f'Do the {name} thing.'}\n",
        encoding="utf-8",
    )


async def _session_with_skills(tmp_path, names: dict[str, str] | None = None):
    root = tmp_path / "skills"
    for name, desc in (names or {"review": "Review code", "docs": "Write docs"}).items():
        _write_skill(root, name, desc)
    session = await _make_session(tmp_path, [_text_turn(["ok"])] * 6)
    session._skills_loader = lambda: load_skills([root])
    session.reload_skills()
    return session, root


class TestListing:
    async def test_lists_loaded_skills(self, tmp_path):
        session, _ = await _session_with_skills(tmp_path)
        out = await session.handle_command("/skills")
        await session.shutdown()
        assert "2 skill(s) loaded" in out
        assert "/skill:review" in out
        assert "Review code" in out

    async def test_empty_state_explains_where_to_put_them(self, tmp_path):
        session = await _make_session(tmp_path, [_text_turn(["ok"])] * 4)
        out = await session.handle_command("/skills")
        await session.shutdown()
        assert "No skills loaded" in out
        assert "SKILL.md" in out

    async def test_show_single_skill(self, tmp_path):
        session, _ = await _session_with_skills(tmp_path)
        out = await session.handle_command("/skills review")
        await session.shutdown()
        assert "review — Review code" in out
        assert "Do the review thing." in out

    async def test_unknown_skill_lists_known_ones(self, tmp_path):
        session, _ = await _session_with_skills(tmp_path)
        out = await session.handle_command("/skills nope")
        await session.shutdown()
        assert "Unknown skill" in out
        assert "review" in out

    @pytest.mark.parametrize("arg", ["review", "/skill:review", "REVIEW"])
    async def test_lookup_is_forgiving(self, tmp_path, arg):
        session, _ = await _session_with_skills(tmp_path)
        assert session.get_skill(arg) is not None
        await session.shutdown()


class TestReload:
    async def test_picks_up_a_new_skill(self, tmp_path):
        session, root = await _session_with_skills(tmp_path)
        assert len(session.skills) == 2

        _write_skill(root, "audit", "Audit dependencies")
        out = await session.handle_command("/skills reload")
        await session.shutdown()
        assert "3 skill(s)" in out

    async def test_reload_reflected_in_listing(self, tmp_path):
        session, root = await _session_with_skills(tmp_path)
        _write_skill(root, "audit", "Audit dependencies")
        await session.handle_command("/skills reload")
        out = await session.handle_command("/skills")
        await session.shutdown()
        assert "/skill:audit" in out

    async def test_general_reload_also_refreshes_skills(self, tmp_path):
        session, root = await _session_with_skills(tmp_path)
        _write_skill(root, "audit", "Audit dependencies")
        await session.reload()
        await session.shutdown()
        assert len(session.skills) == 3


class TestInvocation:
    async def test_expand_skill_builds_invocation_block(self, tmp_path):
        session, _ = await _session_with_skills(tmp_path)
        block = session.expand_skill("/skill:review the parser")
        await session.shutdown()
        assert block is not None
        assert 'name="review"' in block
        assert 'arguments="the parser"' in block
        assert "Do the review thing." in block

    async def test_expand_returns_none_for_plain_text(self, tmp_path):
        session, _ = await _session_with_skills(tmp_path)
        assert session.expand_skill("just a normal prompt") is None
        await session.shutdown()

    async def test_expand_returns_none_for_unknown_skill(self, tmp_path):
        session, _ = await _session_with_skills(tmp_path)
        assert session.expand_skill("/skill:missing args") is None
        await session.shutdown()

    async def test_submit_sends_the_expanded_body(self, tmp_path):
        """The model must receive the skill body, not the raw slash command."""
        session, _ = await _session_with_skills(tmp_path)
        provider = session._provider
        async for _ in session.submit("/skill:review the parser"):
            pass
        _, _, messages, _ = provider.calls[0]
        await session.shutdown()

        from delta_harness.contracts.transcript import surface_text

        sent = " ".join(surface_text(m) for m in messages)
        assert "Do the review thing." in sent
        assert 'name="review"' in sent

    async def test_plain_prompt_is_untouched(self, tmp_path):
        session, _ = await _session_with_skills(tmp_path)
        provider = session._provider
        async for _ in session.submit("hello there"):
            pass
        _, _, messages, _ = provider.calls[0]
        await session.shutdown()

        from delta_harness.contracts.transcript import surface_text

        sent = " ".join(surface_text(m) for m in messages)
        assert "hello there" in sent
        assert "skill-invocation" not in sent


class TestRegistration:
    def test_skills_is_a_known_command(self):
        from delta_app.conversation import CodingSession

        assert any(name == "skills" for name, _ in CodingSession.COMMANDS)

    async def test_autocomplete_offers_it(self, tmp_path):
        pytest.importorskip("textual")
        from delta_app.conversation import CodingSession
        from delta_app.tui.completions import match_commands

        names = [c.name for c in match_commands("/sk", CodingSession.COMMANDS)]
        assert "skills" in names
