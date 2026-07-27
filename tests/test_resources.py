"""Tests for resources.py — canonical paths, resource discovery, and markdown helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from delta_app.resources import (
    DELTA_HOME_ENV,
    DELTA_SESSIONS_DIR_ENV,
    DeltaPaths,
    MarkdownResource,
    ResourceDiagnostic,
    collect_override_diagnostics,
    default_paths,
    derive_description,
    parse_front_matter,
    parse_markdown,
    prompt_search_paths,
    resource_search_paths,
    skill_search_paths,
    theme_search_paths,
)


# ── DeltaPaths: canonical locations ───────────────────────────────────────


class TestDeltaPaths:
    def test_default_home(self) -> None:
        paths = DeltaPaths(home=Path("/home/user/.delta"), project=Path("/project"))
        assert paths.home == Path("/home/user/.delta")

    def test_sessions(self) -> None:
        paths = DeltaPaths(home=Path("/home/user/.delta"), project=Path("/project"))
        assert paths.sessions == Path("/home/user/.delta/sessions")

    def test_logs(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        assert paths.logs == Path("/h/.delta/logs")

    def test_user_skills(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        assert paths.user_skills == Path("/h/.delta/skills")

    def test_user_prompts(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        assert paths.user_prompts == Path("/h/.delta/prompts")

    def test_user_themes(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        assert paths.user_themes == Path("/h/.delta/themes")

    def test_agents_home(self) -> None:
        paths = DeltaPaths(home=Path("/home/user/.delta"), project=Path("/p"))
        assert paths.agents_home == Path("/home/user/.agents")

    def test_agents_skills(self) -> None:
        paths = DeltaPaths(home=Path("/home/user/.delta"), project=Path("/p"))
        assert paths.agents_skills == Path("/home/user/.agents/skills")

    def test_agents_prompts(self) -> None:
        paths = DeltaPaths(home=Path("/home/user/.delta"), project=Path("/p"))
        assert paths.agents_prompts == Path("/home/user/.agents/prompts")

    def test_project_skills(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/my/project"))
        assert paths.project_skills == Path("/my/project/.delta/skills")

    def test_project_prompts(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/my/project"))
        assert paths.project_prompts == Path("/my/project/.delta/prompts")

    def test_project_themes(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/my/project"))
        assert paths.project_themes == Path("/my/project/.delta/themes")

    def test_project_agents_skills(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        assert paths.project_agents_skills == Path("/p/.agents/skills")

    def test_project_agents_prompts(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        assert paths.project_agents_prompts == Path("/p/.agents/prompts")

    def test_frozen(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        with pytest.raises(AttributeError):
            paths.home = Path("/other")  # type: ignore[misc]


# ── DeltaPaths: env var resolution ────────────────────────────────────────


class TestDeltaPathsEnv:
    def test_delta_home_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DELTA_HOME_ENV, "/custom/home")
        paths = default_paths()
        assert paths.home == Path("/custom/home")

    def test_sessions_dir_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DELTA_SESSIONS_DIR_ENV, "/custom/sessions")
        paths = default_paths()
        assert paths.sessions == Path("/custom/sessions")

    def test_sessions_dir_env_overrides_home(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(DELTA_HOME_ENV, "/custom/home")
        monkeypatch.setenv(DELTA_SESSIONS_DIR_ENV, "/other/sessions")
        paths = default_paths()
        assert paths.sessions == Path("/other/sessions")
        assert paths.home == Path("/custom/home")


# ── Skill search paths ───────────────────────────────────────────────────


class TestSkillSearchPaths:
    def test_order_is_highest_first(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        result = skill_search_paths(paths)
        assert result == [
            Path("/p/.agents/skills"),
            Path("/p/.delta/skills"),
            Path("/h/.agents/skills"),
            Path("/h/.delta/skills"),
        ]

    def test_deduplication(self, tmp_path: Path) -> None:
        # When home and project resolve to the same tree, no duplicates
        home = tmp_path / ".delta"
        home.mkdir()
        paths = DeltaPaths(home=home, project=tmp_path)
        result = skill_search_paths(paths)
        # All should be unique (after resolve)
        resolved = [p.resolve() for p in result]
        assert len(resolved) == len(set(resolved))


# ── Prompt search paths ──────────────────────────────────────────────────


class TestPromptSearchPaths:
    def test_order(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        result = prompt_search_paths(paths)
        assert result == [
            Path("/p/.agents/prompts"),
            Path("/p/.delta/prompts"),
            Path("/h/.agents/prompts"),
            Path("/h/.delta/prompts"),
        ]


# ── Theme search paths ───────────────────────────────────────────────────


class TestThemeSearchPaths:
    def test_delta_only(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        result = theme_search_paths(paths)
        assert result == [
            Path("/p/.delta/themes"),
            Path("/h/.delta/themes"),
        ]
        # No .agents directories
        for p in result:
            assert ".agents" not in str(p)


# ── Generic resource search paths ────────────────────────────────────────


class TestResourceSearchPaths:
    def test_full_precedence(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        result = resource_search_paths(paths, "widgets")
        assert result == [
            Path("/p/.agents/widgets"),
            Path("/p/.delta/widgets"),
            Path("/h/.agents/widgets"),
            Path("/h/.delta/widgets"),
        ]

    def test_delta_only_mode(self) -> None:
        paths = DeltaPaths(home=Path("/h/.delta"), project=Path("/p"))
        result = resource_search_paths(paths, "widgets", delta_only=True)
        assert result == [
            Path("/p/.delta/widgets"),
            Path("/h/.delta/widgets"),
        ]


# ── Override diagnostics ─────────────────────────────────────────────────


class TestOverrideDiagnostics:
    def test_no_overlaps(self) -> None:
        discovered = {"a": Path("/high/a")}
        new = {"b": Path("/low/b")}
        diags = collect_override_diagnostics(discovered, new, Path("/low"))
        assert diags == []

    def test_overlap_reported(self) -> None:
        discovered = {"tool": Path("/project/tool")}
        new = {"tool": Path("/user/tool")}
        diags = collect_override_diagnostics(discovered, new, Path("/user"))
        assert len(diags) == 1
        assert diags[0].resource_name == "tool"
        assert diags[0].winner == Path("/project/tool")
        assert diags[0].overridden == Path("/user/tool")
        assert "overridden" in diags[0].message


# ── Frontmatter parsing ──────────────────────────────────────────────────


class TestParseFrontMatter:
    def test_no_frontmatter(self) -> None:
        meta, body = parse_front_matter("Just some text.")
        assert meta == {}
        assert body == "Just some text."

    def test_simple_frontmatter(self) -> None:
        text = "---\nname: test\ndescription: A test\n---\nBody here."
        meta, body = parse_front_matter(text)
        assert meta == {"name": "test", "description": "A test"}
        assert body == "Body here."

    def test_quoted_values(self) -> None:
        text = '---\nname: "quoted"\n---\nBody.'
        meta, body = parse_front_matter(text)
        assert meta["name"] == "quoted"

    def test_single_quoted_values(self) -> None:
        text = "---\nname: 'single'\n---\nBody."
        meta, body = parse_front_matter(text)
        assert meta["name"] == "single"

    def test_comments_skipped(self) -> None:
        text = "---\n# comment\nname: val\n---\nBody."
        meta, body = parse_front_matter(text)
        assert meta == {"name": "val"}

    def test_blank_lines_skipped(self) -> None:
        text = "---\n\nname: val\n\n---\nBody."
        meta, body = parse_front_matter(text)
        assert meta == {"name": "val"}


# ── Description derivation ───────────────────────────────────────────────


class TestDeriveDescription:
    def test_first_paragraph(self) -> None:
        assert derive_description("# Title\n\nFirst paragraph.") == "First paragraph."

    def test_skips_headings(self) -> None:
        assert derive_description("# H1\n## H2\nActual text.") == "Actual text."

    def test_skips_horizontal_rules(self) -> None:
        assert derive_description("---\nText.") == "Text."

    def test_empty_body(self) -> None:
        assert derive_description("") == ""

    def test_only_headings(self) -> None:
        assert derive_description("# H1\n## H2") == ""


# ── MarkdownResource ─────────────────────────────────────────────────────


class TestMarkdownResource:
    def test_name_from_metadata(self) -> None:
        r = MarkdownResource(
            path=Path("/skills/refactor/SKILL.md"),
            body="Body text.",
            metadata={"name": "my-skill"},
        )
        assert r.name == "my-skill"

    def test_name_fallback_to_stem(self) -> None:
        r = MarkdownResource(
            path=Path("/prompts/summarize.md"),
            body="Body.",
            metadata={},
        )
        assert r.name == "summarize"

    def test_description_from_metadata(self) -> None:
        r = MarkdownResource(
            path=Path("/x.md"),
            body="Body.",
            metadata={"description": "Explicit desc"},
        )
        assert r.description == "Explicit desc"

    def test_description_fallback_to_body(self) -> None:
        r = MarkdownResource(
            path=Path("/x.md"),
            body="# Title\n\nFirst line.",
            metadata={},
        )
        assert r.description == "First line."

    def test_metadata_json(self) -> None:
        r = MarkdownResource(
            path=Path("/x.md"),
            body="",
            metadata={"name": "test", "author": "me"},
        )
        result = r.metadata_json()
        assert result == {"name": "test", "author": "me"}

    def test_frozen(self) -> None:
        r = MarkdownResource(path=Path("/x.md"), body="", metadata={})
        with pytest.raises(AttributeError):
            r.body = "changed"  # type: ignore[misc]


# ── parse_markdown ────────────────────────────────────────────────────────


class TestParseMarkdown:
    def test_basic_file(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text("---\nname: hello\n---\n# Greeting\n\nHi there.", encoding="utf-8")
        resource = parse_markdown(md)
        assert resource.name == "hello"
        assert resource.path == md
        assert "Hi there" in resource.body

    def test_no_frontmatter(self, tmp_path: Path) -> None:
        md = tmp_path / "plain.md"
        md.write_text("Just text.", encoding="utf-8")
        resource = parse_markdown(md)
        assert resource.name == "plain"
        assert resource.metadata == {}
        assert resource.body == "Just text."

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            parse_markdown(tmp_path / "missing.md")

    def test_description_derived(self, tmp_path: Path) -> None:
        md = tmp_path / "desc.md"
        md.write_text("# Title\n\nDerived description.", encoding="utf-8")
        resource = parse_markdown(md)
        assert resource.description == "Derived description."


# ── Integration: instructions.py AGENTS.md precedence ─────────────────────


class TestAgentsMdPrecedence:
    def test_project_root_agents_md(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Project-root AGENTS.md is discovered."""
        monkeypatch.setenv(DELTA_HOME_ENV, str(tmp_path / "home" / ".delta"))
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("Project instructions here.", encoding="utf-8")

        from delta_app.instructions import _project_context

        result = _project_context(str(tmp_path))
        assert "Project instructions here." in result

    def test_user_delta_agents_md(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """~/.delta/AGENTS.md is discovered."""
        home = tmp_path / "home" / ".delta"
        home.mkdir(parents=True)
        monkeypatch.setenv(DELTA_HOME_ENV, str(home))
        (home / "AGENTS.md").write_text("User-level context.", encoding="utf-8")

        from delta_app.instructions import _project_context

        result = _project_context(str(tmp_path / "some" / "project"))
        assert "User-level context." in result

    def test_precedence_concatenates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple AGENTS.md files are concatenated."""
        home = tmp_path / "home" / ".delta"
        home.mkdir(parents=True)
        monkeypatch.setenv(DELTA_HOME_ENV, str(home))
        (home / "AGENTS.md").write_text("Global rules.", encoding="utf-8")

        project = tmp_path / "project"
        project.mkdir()
        (project / "AGENTS.md").write_text("Local rules.", encoding="utf-8")

        from delta_app.instructions import _project_context

        result = _project_context(str(project))
        assert "Global rules." in result
        assert "Local rules." in result

    def test_no_agents_md(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No AGENTS.md anywhere returns empty string."""
        monkeypatch.setenv(DELTA_HOME_ENV, str(tmp_path / "home" / ".delta"))

        from delta_app.instructions import _project_context

        result = _project_context(str(tmp_path / "empty"))
        assert result == ""

    def test_project_delta_agents_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """<project>/.delta/AGENTS.md is discovered."""
        monkeypatch.setenv(DELTA_HOME_ENV, str(tmp_path / "home" / ".delta"))
        project = tmp_path / "project"
        delta_dir = project / ".delta"
        delta_dir.mkdir(parents=True)
        (delta_dir / "AGENTS.md").write_text("Delta project context.", encoding="utf-8")

        from delta_app.instructions import _project_context

        result = _project_context(str(project))
        assert "Delta project context." in result

    def test_project_agents_dir_agents_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """<project>/.agents/AGENTS.md is discovered."""
        monkeypatch.setenv(DELTA_HOME_ENV, str(tmp_path / "home" / ".delta"))
        project = tmp_path / "project"
        agents_dir = project / ".agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "AGENTS.md").write_text("Cross-tool project context.", encoding="utf-8")

        from delta_app.instructions import _project_context

        result = _project_context(str(project))
        assert "Cross-tool project context." in result


# ── Integration: sessions under user home ─────────────────────────────────


class TestSessionsUnderHome:
    def test_sessions_default_location(self) -> None:
        paths = DeltaPaths(home=Path("/home/user/.delta"), project=Path("/project"))
        assert paths.sessions == Path("/home/user/.delta/sessions")
        # Sessions are NOT under the project directory
        assert not str(paths.sessions).startswith("/project")

    def test_sessions_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DELTA_SESSIONS_DIR_ENV, "/custom/sessions")
        paths = default_paths()
        assert paths.sessions == Path("/custom/sessions")
