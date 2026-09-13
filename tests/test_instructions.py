"""Tests for system-prompt assembly."""

from __future__ import annotations

from pathlib import Path

from delta_app.instructions import system_prompt
from delta_app.skillset import Skill
from delta_harness.contracts.tooling import ToolOutcome, ToolSpec


async def _noop_run(tool_call_id, arguments, signal=None, on_update=None) -> ToolOutcome:
    return ToolOutcome(content=[])


def _tool(
    name: str = "read",
    *,
    prompt_snippet: str | None = None,
    prompt_guidelines: tuple[str, ...] = (),
) -> ToolSpec:
    return ToolSpec(
        name=name,
        label=name,
        description=f"test {name}",
        parameters={},
        run=_noop_run,
        prompt_snippet=prompt_snippet,
        prompt_guidelines=prompt_guidelines,
    )


def _skill(name: str = "deploy") -> Skill:
    return Skill(
        name=name, description=f"{name} the app", body="body", source=Path(f"{name}/SKILL.md")
    )


class TestBaseIdentity:
    def test_always_present(self, tmp_path: Path) -> None:
        prompt = system_prompt(cwd=str(tmp_path), include_date=False)
        assert "Delta" in prompt

    def test_empty_inputs_produce_no_empty_headers(self, tmp_path: Path) -> None:
        prompt = system_prompt(cwd=str(tmp_path), include_date=False)
        assert "## Tool guidelines" not in prompt
        assert "Available skills" not in prompt
        assert "Project instructions" not in prompt



class TestToolGuidelines:
    def test_tool_with_snippet_and_guidelines_rendered(self, tmp_path: Path) -> None:
        tool = _tool(
            prompt_snippet="Read before you write.",
            prompt_guidelines=("Never guess file contents.",),
        )
        prompt = system_prompt(tools=[tool], cwd=str(tmp_path), include_date=False)
        assert "## Tool guidelines" in prompt
        assert "### read" in prompt
        assert "Read before you write." in prompt
        assert "- Never guess file contents." in prompt

    def test_tool_with_neither_is_omitted(self, tmp_path: Path) -> None:
        tool = _tool()  # no snippet, no guidelines
        prompt = system_prompt(tools=[tool], cwd=str(tmp_path), include_date=False)
        assert "## Tool guidelines" not in prompt

    def test_duplicate_guideline_printed_once(self, tmp_path: Path) -> None:
        shared = "Read before you write."
        prompt = system_prompt(
            tools=[
                _tool("read", prompt_guidelines=(shared,)),
                _tool("write", prompt_guidelines=(shared,)),
            ],
            cwd=str(tmp_path),
            include_date=False,
        )
        assert prompt.count(shared) == 1

    def test_multiple_tools_each_get_a_section(self, tmp_path: Path) -> None:
        tools = [
            _tool("read", prompt_snippet="Read stuff."),
            _tool("write", prompt_snippet="Write stuff."),
        ]
        prompt = system_prompt(tools=tools, cwd=str(tmp_path), include_date=False)
        assert "### read" in prompt
        assert "### write" in prompt


class TestSkillSection:
    def test_skills_render_via_skill_index(self, tmp_path: Path) -> None:
        prompt = system_prompt(skills=[_skill("deploy")], cwd=str(tmp_path), include_date=False)
        assert "Available skills" in prompt
        assert "/skill:deploy" in prompt

    def test_no_skills_no_section(self, tmp_path: Path) -> None:
        prompt = system_prompt(skills=[], cwd=str(tmp_path), include_date=False)
        assert "Available skills" not in prompt


class TestProjectContext:
    def test_agents_md_included_verbatim(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("Run pytest before committing.")
        prompt = system_prompt(cwd=str(tmp_path), include_date=False)
        assert "Project instructions (AGENTS.md)" in prompt
        assert "Run pytest before committing." in prompt

    def test_missing_agents_md_no_section(self, tmp_path: Path) -> None:
        prompt = system_prompt(cwd=str(tmp_path), include_date=False)
        assert "Project instructions" not in prompt

    def test_empty_agents_md_no_section(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("   \n  ")
        prompt = system_prompt(cwd=str(tmp_path), include_date=False)
        assert "Project instructions" not in prompt


class TestEnvironmentSuffix:
    def test_cwd_always_present(self, tmp_path: Path) -> None:
        prompt = system_prompt(cwd=str(tmp_path), include_date=False)
        assert f"Working directory: {tmp_path}" in prompt

    def test_date_included_by_default_flag(self, tmp_path: Path) -> None:
        with_date = system_prompt(cwd=str(tmp_path), include_date=True)
        without_date = system_prompt(cwd=str(tmp_path), include_date=False)
        assert "Current date:" in with_date
        assert "Current date:" not in without_date

    def test_default_cwd_is_getcwd_when_omitted(self) -> None:
        import os

        prompt = system_prompt(include_date=False)
        assert f"Working directory: {os.getcwd()}" in prompt


class TestFullComposition:
    def test_everything_together_in_order(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("Use pytest.")
        tool = _tool(prompt_snippet="Read carefully.")
        prompt = system_prompt(
            tools=[tool], skills=[_skill()], cwd=str(tmp_path), include_date=False
        )
        # sections appear in the documented order
        identity_idx = prompt.index("Delta")
        tools_idx = prompt.index("## Tool guidelines")
        skills_idx = prompt.index("Available skills")
        context_idx = prompt.index("Project instructions")
        env_idx = prompt.index("Working directory")
        assert identity_idx < tools_idx < skills_idx < context_idx < env_idx
