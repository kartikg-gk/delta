"""/system: attributed view of the live system prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from delta_app.conversation import CodingSession
from delta_app.instructions import prompt_sections, system_prompt
from delta_model.scripted import ReplayProvider


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("DELTA_HOME", str(home))
    proj = tmp_path / "proj"
    (proj / ".delta").mkdir(parents=True)
    (home / "AGENTS.md").write_text("user-wide rules", encoding="utf-8")
    (proj / "AGENTS.md").write_text("project rules", encoding="utf-8")
    return proj


async def _session(project: Path, system: str, tmp_path: Path) -> CodingSession:
    return await CodingSession.create(
        provider=ReplayProvider([]), provider_name="test", model="m",
        system=system, sessions_dir=tmp_path / "sessions", cwd=str(project),
    )


def test_sections_join_back_to_the_prompt(project: Path) -> None:
    sections = prompt_sections(cwd=str(project))
    assert "\n\n".join(s.text for s in sections) == system_prompt(cwd=str(project))
    origins = [s.origin for s in sections]
    assert origins[0] == "built-in"
    assert str(project / "AGENTS.md") in origins
    assert any(o.endswith("AGENTS.md") and "home" in o for o in origins)


async def test_system_command_lists_numbered_attributed_sections(
    project: Path, tmp_path: Path,
) -> None:
    session = await _session(project, system_prompt(cwd=str(project)), tmp_path)
    before = len(session.transcript)
    output = await session.handle_command("/system")

    assert output is not None
    assert "── 1. Identity  [built-in]" in output
    assert f"[{project / 'AGENTS.md'}]" in output
    assert "project rules" in output
    assert len(session.transcript) == before  # display only, not model context
    await session.shutdown()


async def test_unreconstructable_prompt_falls_back_to_one_section(
    project: Path, tmp_path: Path,
) -> None:
    session = await _session(project, "Custom override prompt.", tmp_path)
    sections = session.system_sections()
    assert len(sections) == 1
    assert sections[0].origin == "runtime-composed"
    assert sections[0].text == "Custom override prompt."
    await session.shutdown()


async def test_plan_mode_block_is_attributed(project: Path, tmp_path: Path) -> None:
    session = await _session(project, system_prompt(cwd=str(project)), tmp_path)
    session.set_plan_mode(True)
    sections = session.system_sections()
    assert sections[-1].title == "Plan mode"
    assert sections[0].origin == "built-in"
    await session.shutdown()
